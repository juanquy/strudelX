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
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

MODEL_NAME = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")

print(f"📦 Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"🧠 Loading base weights for {MODEL_NAME} on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="cpu"
)
model.eval()

SYSTEM_PROMPT = """You are Strudel AI, a master live-coding music assistant and algorithmic pattern engineer.
You help music producers and live-coders write beautiful, rhythmically intricate, and syntactically correct Strudel patterns.

Guidelines:
1. Always output valid Strudel JavaScript code directly.
2. Use expressive mini-notation (e.g., s("bd [sd ~] [hh*4]"), struct("t(5,8)")).
3. Use rich built-in sound banks (e.g. .bank("tr909"), .bank("tr808"), sawtooth, triangle, sine).
4. Prioritize clean, runnable Strudel JavaScript."""


@spaces.GPU(duration=60)
def generate_strudel_code(prompt, current_code="", genre="Progressive House", bpm=130, temperature=0.7, max_tokens=1024):
    """ZeroGPU accelerated generation function."""
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


# Read built static index.html content
DIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "website", "dist"))
INDEX_PATH = os.path.join(DIST_DIR, "index.html")

raw_html = ""
if os.path.exists(INDEX_PATH):
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        raw_html = f.read()

custom_css = """
body, .gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100vw !important;
    width: 100vw !important;
    height: 100vh !important;
    overflow: hidden !important;
    background: #0b0b0d !important;
}
footer { display: none !important; }
"""

with gr.Blocks(title="Strudel AI Studio", theme=gr.themes.Monochrome(), css=custom_css) as demo:
    if raw_html:
        gr.HTML(raw_html)
    else:
        gr.Markdown("# Strudel REPL Loading...")

    with gr.Row(visible=False):
        prompt_in = gr.Textbox()
        code_in = gr.Textbox()
        genre_in = gr.Textbox()
        bpm_in = gr.Number()
        temp_in = gr.Number()
        tokens_in = gr.Number()
        out_code = gr.Textbox()
        gen_btn = gr.Button("api_run")
        gen_btn.click(
            fn=generate_strudel_code,
            inputs=[prompt_in, code_in, genre_in, bpm_in, temp_in, tokens_in],
            outputs=out_code,
            api_name="generate_pattern"
        )

app = gr.routes.App.create_app(demo)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(DIST_DIR):
    for folder in ["_astro", "fonts", "icons", "img", "pwa", "bakery", "learn", "workshop", "technical-manual", "recipes"]:
        subpath = os.path.join(DIST_DIR, folder)
        if os.path.exists(subpath):
            app.mount(f"/{folder}", StaticFiles(directory=subpath), name=folder)

    @app.post("/api/generate_pattern")
    async def api_generate_pattern(req: Request):
        try:
            data = await req.json()
            prompt = data.get("prompt", "")
            current_code = data.get("current_code", "")
            genre = data.get("genre", "Progressive House")
            bpm = data.get("bpm", 130)
            code = generate_strudel_code(prompt, current_code, genre, bpm)
            return JSONResponse({"code": code, "data": [code]})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
