# Strudel AI: YouTube Ingestion & Fine-Tuning Pipeline

This toolkit enables downloading Strudel/Tidal live-coding tutorials from YouTube, extracting spoken explanations and editor code keyframes, validating the syntax, fine-tuning **Qwen 2.5 Coder (7B/14B)**, and deploying the model to **Hugging Face Spaces**.

---

## 1. Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

*(Optional: Install `tesseract-ocr` on macOS via `brew install tesseract` or on Ubuntu via `sudo apt-get install tesseract-ocr`).*

---

## 2. Ingest Curated YouTube Channels & Tutorials

You can ingest all target creators (**SwitchAngel, PEPPIN!, Groovin in G, DJ_Dave, Lucy Cheesman, Sound Codex, Shovon Saha**) in one command:

```bash
python batch_ingest.py \
  --config "curated_channels.json" \
  --out-dir "./master_dataset" \
  --whisper-model "base" \
  --max-videos-per-channel 10 \
  --interval 3.0
```

This will automatically discover and download the videos, transcribe spoken techniques via Whisper, extract code states from screen frames via OCR, and build a unified `strudel_master_dataset.jsonl`.

Or run single video ingestion:

```bash
python ingest_youtube.py \
  --url "https://www.youtube.com/watch?v=ZCcpWzhekEY" \
  --out "./dataset_out" \
  --whisper-model "base" \
  --interval 3.0
```

### What this does:
1. **Audio Transcription**: Uses OpenAI Whisper to extract word-level and phrase-level timestamps of the tutor's explanations.
2. **Editor Keyframe Sampling**: Samples video frames when the screen/code changes.
3. **OCR / Vision Extraction**: Extracts clean Strudel mini-notation and JavaScript code from the editor frames.
4. **Dataset Synthesis**: Generates `strudel_training_dataset.jsonl` matching instruction prompts to validated Strudel code.

---

## 3. Fine-Tune Qwen 2.5 Coder with QLoRA

Run `train_lora_qwen.py` on your dataset:

```bash
python train_lora_qwen.py \
  --model-name "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --dataset "./dataset_out/strudel_training_dataset.jsonl" \
  --output-dir "./strudel-qwen-lora" \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-4
```

---

## 4. Deploy to Hugging Face Spaces

1. Create a new **Hugging Face Space** (choose **FastAPI** or **Docker** with GPU).
2. Upload `hf_space_app.py`, `requirements.txt`, and your trained `strudel-qwen-lora/` weights directory.
3. The Space will automatically serve `/api/generate` and `/api/chat` with SSE streaming responses for your Strudel REPL!
