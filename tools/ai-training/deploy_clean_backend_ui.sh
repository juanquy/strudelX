#!/bin/bash
# Deploy pure, clean Gradio Backend UI to Hugging Face Space
set -e
echo "🚀 Deploying 100% clean Gradio AI Backend UI to Hugging Face Space..."

git checkout -f main
git branch -D hf-clean 2>/dev/null || true
git checkout --orphan hf-clean
git rm -rf --cached . 2>/dev/null || true
git reset 2>/dev/null || true

# Add ONLY pure backend files
git add app.py README.md requirements.txt 2>/dev/null || true

git commit -m "Deploy clean Gradio AI Backend & Training Studio UI"
git push space hf-clean:main --force

git checkout -f main
git branch -D hf-clean 2>/dev/null || true
echo "✅ Successfully deployed clean Gradio AI Backend UI to Hugging Face Space!"
