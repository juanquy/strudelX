# IMPORTANT: 'import spaces' MUST be the very first line for Hugging Face ZeroGPU
try:
    import spaces
    HAS_ZEROGPU = True
except ImportError:
    HAS_ZEROGPU = False
    class spaces:
        @staticmethod
        def GPU(fn=None, duration=60):
            def decorator(func):
                return func
            if fn is not None:
                return fn
            return decorator

import os
import re
import torch
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

MODEL_NAME = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
LORA_PATH = os.getenv("LORA_PATH", "./strudel-qwen-lora")

print(f"📦 Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"🧠 Loading base weights for {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

if os.path.exists(LORA_PATH):
    print(f"🔗 Loading LoRA adapter from {LORA_PATH}...")
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model = model.merge_and_unload()
    except Exception as e:
        print(f"⚠️ LoRA load notice: {e}")

model.eval()

SYSTEM_PROMPT = """You are Strudel AI, a master live-coding music assistant and algorithmic pattern engineer.
You help music producers and live-coders write beautiful, rhythmically intricate, and syntactically correct Strudel patterns.

Guidelines:
1. Always output valid Strudel JavaScript code directly.
2. Use expressive mini-notation (e.g., s("bd [sd ~] [hh*4]"), struct("t(5,8)")).
3. Use rich built-in sound banks (e.g. .bank("tr909"), .bank("tr808"), sawtooth, triangle, sine).
4. Prioritize clean, runnable Strudel JavaScript."""

app = FastAPI(title="Strudel REPL & AI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@spaces.GPU(duration=60)
def generate_code_fn(prompt, current_code="", genre="Progressive House", bpm=130, temperature=0.7, max_tokens=1024):
    if torch.cuda.is_available():
        model.to("cuda")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if current_code and str(current_code).strip():
        messages.append({
            "role": "user",
            "content": f"Current Strudel code:\n```javascript\n{str(current_code).strip()}\n```\n\nTask: {prompt}\nGenre: {genre}, Tempo: {bpm} BPM."
        })
    else:
        messages.append({
            "role": "user",
            "content": f"Create a complete Strudel live coding pattern for genre: {genre} at {bpm} BPM.\nPrompt: {prompt}"
        })

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=int(max_tokens),
        temperature=max(0.01, float(temperature)),
        top_p=0.9,
        do_sample=True,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    accumulated_text = ""
    for new_text in streamer:
        accumulated_text += new_text

    thread.join()

    m = re.search(r'```(?:javascript|js)?([\s\S]*?)```', accumulated_text)
    if m:
        return m.group(1).strip()
    return accumulated_text.strip()


@app.post("/api/generate_pattern")
async def api_generate_pattern(req: Request):
    try:
        data = await req.json()
        prompt = data.get("prompt", "")
        current_code = data.get("current_code", "")
        genre = data.get("genre", "Progressive House")
        bpm = data.get("bpm", 130)
        code = generate_code_fn(prompt, current_code, genre, bpm)
        return JSONResponse({"code": code, "data": [code]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/generate")
async def api_generate(req: Request):
    try:
        data = await req.json()
        params = data.get("data", [])
        prompt = params[0] if len(params) > 0 else ""
        current_code = params[1] if len(params) > 1 else ""
        genre = params[2] if len(params) > 2 else "Progressive House"
        bpm = params[3] if len(params) > 3 else 130
        code = generate_code_fn(prompt, current_code, genre, bpm)
        return JSONResponse({"data": [code], "code": code})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


DIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "website", "dist"))

if os.path.exists(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
