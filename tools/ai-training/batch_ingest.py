#!/usr/bin/env python3
"""
batch_ingest.py — Batch Ingestion & Dataset Aggregator for Curated Strudel Channels
Downloads and extracts speech transcripts + editor code frames across all target creators
and merges them into a clean, deduplicated, verified master training dataset.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from ingest_youtube import (
    download_youtube_video,
    transcribe_audio,
    extract_editor_keyframes,
    build_instruction_dataset
)


def get_channel_videos(channel_url: str, max_videos: int = 15) -> list:
    """Fetch video URLs from a YouTube channel using yt-dlp."""
    print(f"🔍 Discovering videos for {channel_url} (max {max_videos})...")
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "url",
        "--playlist-end", str(max_videos),
        channel_url
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        urls = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        return urls
    except Exception as e:
        print(f"⚠️ Error fetching channel videos: {e}")
        return [channel_url] if "watch?v=" in channel_url else []


def main():
    parser = argparse.ArgumentParser(description="Batch ingest curated Strudel YouTube channels.")
    parser.add_argument("--config", type=str, default="curated_channels.json", help="Path to curated channels JSON")
    parser.add_argument("--out-dir", type=str, default="./master_dataset", help="Master output directory")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--max-videos-per-channel", type=int, default=10, help="Max videos per channel")
    parser.add_argument("--interval", type=float, default=3.0, help="Frame sample interval (seconds)")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        channels = json.load(f)

    master_dataset_file = out_dir / "strudel_master_dataset.jsonl"
    all_dataset_items = []

    print(f"🚀 Starting batch ingestion across {len(channels)} curated creators/videos...")

    for i, target in enumerate(channels, 1):
        name = target["name"]
        url = target["url"]
        print(f"\n=========================================")
        print(f"[{i}/{len(channels)}] Processing target: {name} ({target['category']})")
        print(f"URL: {url}")
        print(f"=========================================")

        target_dir = out_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)

        if "watch?v=" in url:
            video_urls = [url]
        else:
            video_urls = get_channel_videos(url, max_videos=args.max_videos_per_channel)

        for v_idx, v_url in enumerate(video_urls, 1):
            video_out_dir = target_dir / f"video_{v_idx}"
            video_dataset_file = video_out_dir / "dataset.jsonl"

            if video_dataset_file.exists():
                print(f"⏩ Already ingested {v_url}, reading existing dataset...")
                with open(video_dataset_file, "r", encoding="utf-8") as vf:
                    for line in vf:
                        if line.strip():
                            all_dataset_items.append(json.loads(line))
                continue

            try:
                print(f"\n▶ Ingesting video {v_idx}/{len(video_urls)}: {v_url}")
                files = download_youtube_video(v_url, video_out_dir)
                transcripts = transcribe_audio(files["audio"], model_size=args.whisper_model)
                keyframes = extract_editor_keyframes(files["video"], video_out_dir, sample_interval_sec=args.interval)
                build_instruction_dataset(transcripts, keyframes, video_dataset_file)

                with open(video_dataset_file, "r", encoding="utf-8") as vf:
                    for line in vf:
                        if line.strip():
                            all_dataset_items.append(json.loads(line))
            except Exception as err:
                print(f"⚠️ Failed to process {v_url}: {err}")

    # Deduplicate and aggregate into master dataset
    print(f"\n💾 Aggregating {len(all_dataset_items)} total patterns into master dataset...")
    seen_codes = set()
    deduped_items = []
    for item in all_dataset_items:
        code_sig = item.get("output", "").strip()
        if code_sig and code_sig not in seen_codes:
            seen_codes.add(code_sig)
            deduped_items.append(item)

    with open(master_dataset_file, "w", encoding="utf-8") as mf:
        for item in deduped_items:
            mf.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"🎉 Master dataset successfully created: {master_dataset_file} ({len(deduped_items)} unique instruction pairs)!")


if __name__ == "__main__":
    main()
