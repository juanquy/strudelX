#!/bin/bash
# Push full Strudel REPL to Hugging Face Docker Space
set -e
echo "🚀 Pushing full Strudel Web REPL with AI Copilot & DAW Studio to Hugging Face Space..."
git branch -D hf-space 2>/dev/null || true
git checkout --orphan hf-space
git reset
git add .
git reset docs/iclc2023-paper/
git commit -m "Deploy full official Strudel Web REPL with AI Copilot & DAW Studio"
git push space hf-space:main --force
git checkout -f main
git branch -D hf-space 2>/dev/null || true
echo "✅ Successfully deployed full Strudel Web REPL to Hugging Face Space!"
