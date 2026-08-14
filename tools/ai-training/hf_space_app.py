"""
hf_space_app.py — Hugging Face Spaces FastAPI & Web UI for Strudel AI Assistant
Provides streaming code generation and live pattern assistance for the Strudel REPL.
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

app = FastAPI(title="Strudel AI Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
LORA_PATH = os.getenv("LORA_PATH", "./strudel-qwen-lora")

print(f"📦 Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"🧠 Loading base model {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)

if os.path.exists(LORA_PATH):
    print(f"🔗 Loading LoRA weights from {LORA_PATH}...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()

model.eval()

SYSTEM_PROMPT = """You are Strudel AI, a master live-coding music assistant and algorithmic pattern engineer.
You help music producers and live-coders write beautiful, rhythmically intricate, and syntactically correct Strudel patterns.

Guidelines:
1. Always output valid Strudel JavaScript code.
2. Use expressive mini-notation (e.g., s("bd [sd ~] [hh*4]"), struct("t(5,8)")).
3. Support 13-channel DAW / Bitwig routing (.midichan(1..13), .midi('IAC Driver')).
4. Explain algorithmic ideas concisely with sound design parameters."""


class GenerateRequest(BaseModel):
    prompt: str
    current_code: Optional[str] = ""
    genre: Optional[str] = "techno"
    bpm: Optional[int] = 130
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Strudel AI Assistant API",
        "device": device,
        "model": MODEL_NAME
    }


@app.post("/api/generate")
async def generate_code(req: GenerateRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if req.current_code:
        messages.append({
            "role": "user",
            "content": f"Current Strudel code:\n```javascript\n{req.current_code}\n```\n\nTask: {req.prompt}"
        })
    else:
        messages.append({
            "role": "user",
            "content": f"Create a Strudel pattern for genre: {req.genre} at {req.bpm} BPM.\nPrompt: {req.prompt}"
        })

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=0.9,
        do_sample=True,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    async def token_generator():
        for text in streamer:
            yield f"data: {text}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
