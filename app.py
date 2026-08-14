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
import torch
import gradio as gr
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
1. Always output valid Strudel JavaScript code.
2. Use expressive mini-notation (e.g., s("bd [sd ~] [hh*4]"), struct("t(5,8)")).
3. Support 13-channel DAW / Bitwig routing (.midichan(1..13), .midi('IAC Driver')).
4. Explain algorithmic ideas concisely with sound design parameters."""


@spaces.GPU(duration=60)
def generate_strudel_code(prompt, current_code, genre, bpm, temperature, max_tokens):
    """ZeroGPU accelerated generation function."""
    if torch.cuda.is_available():
        model.to("cuda")

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
        yield accumulated_text

    thread.join()


custom_css = """
body, .gradio-container {
    background-color: #0b0b0d !important;
    color: #e8e6e1 !important;
    margin: 0 !important;
    padding: 10px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
iframe {
    border: 1px solid #232328;
    border-radius: 8px;
    background: #000;
}
"""

with gr.Blocks(title="Strudel AI Studio — Live REPL & ZeroGPU Copilot", theme=gr.themes.Monochrome(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 🎵 Strudel AI Studio
        ### Live Coding REPL with Integrated ZeroGPU AI Copilot & 13-Channel Bitwig Arranger
        """
    )

    with gr.Row():
        # Left side: Live Interactive Strudel REPL
        with gr.Column(scale=3):
            gr.Markdown("### 🎹 Interactive Strudel REPL Editor & Live Sound Engine")
            strudel_iframe = gr.HTML(
                """
                <iframe
                    id="strudel-repl"
                    src="https://strudel.cc"
                    width="100%"
                    height="720px"
                    allow="midi; microphone; audio-capture"
                    style="border: 1px solid #333338; border-radius: 8px;"
                ></iframe>
                """
            )

        # Right side: AI Copilot & DAW Studio
        with gr.Column(scale=2):
            gr.Markdown("### ✨ AI Copilot (ZeroGPU A100)")
            prompt_input = gr.Textbox(
                label="Prompt / Musical Idea",
                placeholder="e.g. Create a 130 BPM progressive house arrangement with Euclidean bass and 13-channel MIDI routing for Bitwig",
                lines=3,
            )
            code_input = gr.Textbox(
                label="Current Pattern Code (Optional)",
                placeholder="Paste code here to transform or remix...",
                lines=3,
            )

            with gr.Row():
                genre_dropdown = gr.Dropdown(
                    label="Genre",
                    choices=["Progressive House", "Techno", "Trance", "Drum & Bass", "Dubstep", "Deep House", "UK Garage / Breakbeat", "Ambient"],
                    value="Progressive House"
                )
                bpm_input = gr.Number(label="BPM", value=130, precision=0)

            with gr.Accordion("Model Settings", open=False):
                temp_slider = gr.Slider(minimum=0.1, maximum=1.2, value=0.7, step=0.05, label="Temperature")
                tokens_slider = gr.Slider(minimum=256, maximum=2048, value=1024, step=128, label="Max Tokens")

            generate_btn = gr.Button("🚀 Generate Strudel Pattern", variant="primary")

            output_code = gr.Code(
                label="AI Generated Code (Copy & Paste to Live REPL)",
                language="javascript",
                lines=12,
            )

    generate_btn.click(
        fn=generate_strudel_code,
        inputs=[prompt_input, code_input, genre_dropdown, bpm_input, temp_slider, tokens_slider],
        outputs=output_code,
        api_name="generate"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
