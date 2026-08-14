#!/usr/bin/env python3
"""
ingest_youtube.py — Multimodal Strudel Tutorial Ingestion & Dataset Generator
Extracts speech transcripts (Whisper) and editor code frames (Vision/OCR) from YouTube tutorials
to produce validated instruction-tuning datasets for Qwen 2.5 Coder.
"""

import os
import sys
import json
import argparse
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:
    pass

try:
    import whisper
except ImportError:
    pass


def run_cmd(cmd: List[str]) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr}")
    return res.stdout


def download_youtube_video(url: str, output_dir: Path) -> Dict[str, Path]:
    """Download video and audio using yt-dlp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_template = str(output_dir / "video.%(ext)s")
    audio_path = output_dir / "audio.wav"

    print(f"📥 Downloading video and audio from {url}...")
    # Download best video + audio
    run_cmd(["yt-dlp", "-f", "bestvideo+bestaudio/best", "--merge-output-format", "mp4", "-o", video_template, url])

    video_path = output_dir / "video.mp4"
    if not video_path.exists():
        # Find whichever extension yt-dlp produced
        mp4s = list(output_dir.glob("video.*"))
        if mp4s:
            video_path = mp4s[0]

    # Extract 16kHz mono audio for Whisper
    print("🎵 Extracting 16kHz WAV audio for Whisper...")
    run_cmd(["ffmpeg", "-y", "-i", str(video_path), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(audio_path)])

    return {"video": video_path, "audio": audio_path}


def transcribe_audio(audio_path: Path, model_size: str = "base") -> List[Dict[str, Any]]:
    """Transcribe audio with Whisper to extract timestamped segments."""
    print(f"🎙️ Transcribing audio with Whisper ({model_size})...")
    model = whisper.load_model(model_size)
    result = model.transcribe(str(audio_path), verbose=False)
    
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })
    print(f"✅ Transcribed {len(segments)} spoken segments.")
    return segments


def extract_editor_keyframes(video_path: Path, output_dir: Path, sample_interval_sec: float = 3.0) -> List[Dict[str, Any]]:
    """Sample keyframe images where editor content changes."""
    print(f"🎞️ Sampling video frames every {sample_interval_sec}s...")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(fps * sample_interval_sec)

    frame_idx = 0
    saved_frames = []
    last_frame_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Check for significant scene / code change
            is_different = True
            if last_frame_gray is not None:
                diff = cv2.absdiff(gray, last_frame_gray)
                score = np.mean(diff)
                if score < 2.5:  # minimal change
                    is_different = False

            if is_different:
                frame_filename = frames_dir / f"frame_{int(timestamp):05d}.png"
                cv2.imwrite(str(frame_filename), frame)
                saved_frames.append({
                    "timestamp": timestamp,
                    "image_path": frame_filename
                })
                last_frame_gray = gray

        frame_idx += 1

    cap.release()
    print(f"✅ Extracted {len(saved_frames)} keyframes.")
    return saved_frames


def ocr_code_from_frame(image_path: Path) -> Optional[str]:
    """Extract code text from frame via OCR / Tesseract or regex heuristic."""
    try:
        import pytesseract
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        # Filter for Strudel/JS patterns
        if any(keyword in text for keyword in ["s(", "note(", "sound(", "arrange(", "stack(", "setcpm(", "jux(", "every("]):
            return clean_strudel_code(text)
    except Exception:
        pass
    return None


def clean_strudel_code(raw_text: str) -> str:
    """Clean OCR artifacts and normalize Strudel pattern code."""
    lines = raw_text.splitlines()
    clean_lines = []
    for line in lines:
        l = line.strip()
        if not l:
            continue
        # Remove common OCR noise
        l = re.sub(r'^[>\$#|•\s]+', '', l)
        clean_lines.append(l)
    return "\n".join(clean_lines)


def build_instruction_dataset(transcripts: List[Dict[str, Any]], keyframes: List[Dict[str, Any]], output_file: Path):
    """Align timestamped speech explanations with corresponding code frames into JSONL."""
    print("🧩 Aligning spoken transcripts with code states...")
    dataset = []

    for frame in keyframes:
        t = frame["timestamp"]
        code = ocr_code_from_frame(frame["image_path"])
        if not code:
            continue

        # Find speech explanation in surrounding window [t-10s, t+10s]
        relevant_speech = [
            seg["text"] for seg in transcripts
            if abs(seg["start"] - t) < 12.0 or abs(seg["end"] - t) < 12.0
        ]
        explanation = " ".join(relevant_speech).strip()
        if not explanation:
            explanation = "Live coding pattern with Strudel mini-notation and WebAudio synthesis"

        instruction_item = {
            "instruction": f"Write a Strudel live-coding pattern for: {explanation}",
            "input": "",
            "output": code,
            "metadata": {
                "timestamp": t,
                "frame": str(frame["image_path"])
            }
        }
        dataset.append(instruction_item)

    print(f"💾 Saving {len(dataset)} instruction pairs to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"🎉 Dataset ready: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Ingest YouTube Strudel tutorials for AI training.")
    parser.add_argument("--url", type=str, required=True, help="YouTube video or playlist URL")
    parser.add_argument("--out", type=str, default="./dataset_out", help="Output directory")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--interval", type=float, default=3.0, help="Frame sampling interval in seconds")

    args = parser.parse_args()
    out_dir = Path(args.out)

    files = download_youtube_video(args.url, out_dir)
    transcripts = transcribe_audio(files["audio"], model_size=args.whisper_model)
    keyframes = extract_editor_keyframes(files["video"], out_dir, sample_interval_sec=args.interval)
    build_instruction_dataset(transcripts, keyframes, out_dir / "strudel_training_dataset.jsonl")


if __name__ == "__main__":
    main()
