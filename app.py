"""
app.py — Hugging Face ZeroGPU Gradio App for Strudel AI Assistant
"""

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

# Try importing spaces for Hugging Face ZeroGPU
try:
    import spaces
    HAS_ZEROGPU = True
except ImportError:
    HAS_ZEROGPU = False
    class spaces:
        @staticmethod
        def GPU(fn):
            return fn

MODEL_NAME = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
LORA_PATH = os.getenv("LORA_PATH", "./strudel-qwen-lora")

print(f"📦 Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print(f"🧠 Loading model {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)

if os.path.exists(LORA_PATH):
    print(f"🔗 Merging LoRA adapter from {LORA_PATH}...")
    try:
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model = model.merge_and_unload()
    except Exception as e:
        print(f"⚠️ LoRA load warning: {e}")

model.eval()

SYSTEM_PROMPT = """You are Strudel AI, a master live-coding music assistant and algorithmic pattern engineer.
You help music producers and live-coders write beautiful, rhythmically intricate, and syntactically correct Strudel patterns.

Guidelines:
1. Always output valid Strudel JavaScript code.
2. Use expressive mini-notation (e.g., s("bd [sd ~] [hh*4]"), struct("t(5,8)")).
3. Support 13-channel DAW / Bitwig routing (.midichan(1..13), .midi('IAC Driver')).
4. Explain algorithmic ideas concisely with sound design parameters."""


@spaces.GPU(duration=60)
def generate_strudel_code(prompt, current_code, genre, bpm, temperature, max_tokens):
    """ZeroGPU accelerated generation function."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if current_code and current_code.strip():
        messages.append({
            "role": "user",
            "content": f"Current Strudel code:\n```javascript\n{current_code.strip()}\n```\n\nTask: {prompt}"
        })
    else:
        messages.append({
            "role": "user",
            "content": f"Create a Strudel pattern for genre: {genre} at {bpm} BPM.\nPrompt: {prompt}"
        })

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=0.9,
        do_sample=True,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    accumulated_text = ""
    for new_text in streamer:
        accumulated_text += new_text
        yield accumulated_text


EXAMPLES = [
    [
        "Create a driving 130 BPM progressive house arrangement with Euclidean bass, punchcard drums, and IAC Driver MIDI routing across channels 1-13.",
        "",
        "Progressive House",
        130,
        0.7,
        1024
    ],
    [
        "Generate a dark, hypnotic techno loop with an acid bassline modulated by a slow LPF sine wave and 16th-note hi-hats.",
        "",
        "Techno",
        132,
        0.7,
        1024
    ],
    [
        "Transform this simple 4-on-the-floor beat into a syncopated UK garage / 2-step groove with swing.",
        's("bd*4").gain(1)\ns("~ sd ~ sd").gain(0.8)\ns("hh*8").gain(0.5)',
        "UK Garage / Breakbeat",
        134,
        0.6,
        1024
    ],
]

custom_css = """
.gradio-container {
    background-color: #0b0b0d;
    color: #e8e6e1;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.gr-button-primary {
    background: #3b8eff !important;
    border: none !important;
}
"""

with gr.Blocks(title="Strudel AI Studio — ZeroGPU Assistant", theme=gr.themes.Monochrome(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 🎵 Strudel AI Studio
        ### Live Coding Music Copilot & 13-Channel Bitwig Arranger (Powered by Hugging Face ZeroGPU)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt / Musical Idea",
                placeholder="e.g. Create a rolling techno bass with Euclidean rhythm and 13-channel MIDI output for Bitwig",
                lines=3,
            )
            code_input = gr.Textbox(
                label="Current Code (Optional, for editing / transforming)",
                placeholder="Paste existing Strudel code here to modify it...",
                lines=4,
            )
            with gr.Row():
                genre_dropdown = gr.Dropdown(
                    label="Genre",
                    choices=["Progressive House", "Techno", "Trance", "Drum & Bass", "Dubstep", "Deep House", "UK Garage / Breakbeat", "Ambient"],
                    value="Progressive House"
                )
                bpm_input = gr.Number(label="BPM", value=130, precision=0)

            with gr.Accordion("Advanced Generation Settings", open=False):
                temp_slider = gr.Slider(minimum=0.1, maximum=1.2, value=0.7, step=0.05, label="Temperature")
                tokens_slider = gr.Slider(minimum=256, maximum=2048, value=1024, step=128, label="Max Tokens")

            generate_btn = gr.Button("🚀 Generate Strudel Pattern", variant="primary")

        with gr.Column(scale=1):
            output_code = gr.Code(
                label="Generated Strudel Code",
                language="javascript",
                lines=20,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt_input, code_input, genre_dropdown, bpm_input, temp_slider, tokens_slider],
        outputs=output_code,
        fn=generate_strudel_code,
        cache_examples=False,
    )

    generate_btn.click(
        fn=generate_strudel_code,
        inputs=[prompt_input, code_input, genre_dropdown, bpm_input, temp_slider, tokens_slider],
        outputs=output_code,
        api_name="generate"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
