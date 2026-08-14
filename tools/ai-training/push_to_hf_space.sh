#!/bin/bash
# Push clean Space bundle to Hugging Face Space
set -e
echo "🚀 Pushing Strudel AI Gradio ZeroGPU app to Hugging Face Space..."
git branch -D hf-space 2>/dev/null || true
git checkout --orphan hf-space
git reset
git add app.py requirements.txt README.md tools/ai-training/
git commit -m "Update Strudel AI ZeroGPU Gradio Space with README metadata"
git push space hf-space:main --force
git checkout -f main
git branch -D hf-space 2>/dev/null || true
echo "✅ Successfully deployed to Hugging Face Space!"
