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
import json
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

MODEL_NAME = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
LORA_PATH = os.getenv("LORA_PATH", "./strudel-qwen-lora")
DATASET_PATH = os.getenv("DATASET_PATH", "./data/strudel_dataset.jsonl")

os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

_tokenizer = None
_model = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        print(f"📦 Loading tokenizer for {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer

def get_model():
    global _model
    if _model is None:
        print(f"🧠 Lazy loading base weights for {MODEL_NAME}...")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        if os.path.exists(LORA_PATH):
            try:
                from peft import PeftModel
                _model = PeftModel.from_pretrained(_model, LORA_PATH)
                _model = _model.merge_and_unload()
            except Exception as e:
                print(f"⚠️ LoRA load notice: {e}")
        _model.eval()
    return _model

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
    tokenizer = get_tokenizer()
    model = get_model()

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


def add_to_dataset(instruction, strudel_code, genre, bpm):
    """Add a new curated Strudel code pattern pair to the training dataset."""
    if not instruction.strip() or not strudel_code.strip():
        return "❌ Error: Prompt and Strudel Code cannot be empty."

    entry = {
        "instruction": f"Create a Strudel pattern for genre: {genre} at {bpm} BPM. {instruction.strip()}",
        "input": "",
        "output": strudel_code.strip(),
        "metadata": {"genre": genre, "bpm": bpm}
    }

    with open(DATASET_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return f"✅ Saved sample to {DATASET_PATH}! Total entries: {get_dataset_count()}"


def get_dataset_count():
    if not os.path.exists(DATASET_PATH):
        return 0
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


@spaces.GPU(duration=300)
def run_lora_finetuning(epochs, batch_size, learning_rate, lora_r):
    """ZeroGPU accelerated LoRA Fine-Tuning Studio."""
    count = get_dataset_count()
    if count < 1:
        return f"❌ Fine-tuning requires at least 1 curated dataset sample in {DATASET_PATH}. Add samples in Dataset Curator tab first!"

    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
        from transformers import TrainingArguments

        print(f"🔥 Starting ZeroGPU LoRA Fine-Tuning on {count} dataset samples...")

        tokenizer = get_tokenizer()
        model = get_model()

        train_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

        def format_chat(sample):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["output"]}
            ]
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

        train_dataset = train_dataset.map(format_chat)

        if torch.cuda.is_available():
            model.to("cuda")

        lora_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_r * 2),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        peft_model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=LORA_PATH,
            num_train_epochs=int(epochs),
            per_device_train_batch_size=int(batch_size),
            gradient_accumulation_steps=2,
            learning_rate=float(learning_rate),
            fp16=torch.cuda.is_available(),
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=1024,
            tokenizer=tokenizer,
            args=training_args,
        )

        trainer.train()
        trainer.model.save_pretrained(LORA_PATH)
        tokenizer.save_pretrained(LORA_PATH)

        return f"🎉 Success! Fine-tuned LoRA model saved to {LORA_PATH}! All future local StrudelX AI queries will use the updated model."
    except Exception as e:
        return f"⚠️ Fine-tuning error: {str(e)}"


custom_css = """
body, .gradio-container {
    background-color: #0b0b0d;
    color: #e8e6e1;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.gr-button-primary {
    background: #3b8eff !important;
    border: none !important;
}
"""

with gr.Blocks(title="Strudel AI Studio & Training Backend", theme=gr.themes.Monochrome(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 🎵 Strudel AI Cloud Backend & Fine-Tuning Studio
        ### ZeroGPU Accelerated API for Local StrudelX Desktop & Browser Apps
        """
    )

    with gr.Tabs():
        with gr.TabItem("🎵 AI Music Generation Sandbox"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_in = gr.Textbox(label="Prompt / Musical Idea", placeholder="e.g. Create a rolling progressive house bassline with 130 BPM", lines=3)
                    code_in = gr.Textbox(label="Current Strudel Code (Optional)", placeholder="Paste existing code to modify...", lines=4)
                    genre_in = gr.Dropdown(label="Genre", choices=["Progressive House", "Techno", "Trance", "Drum & Bass", "Dubstep", "Ambient"], value="Progressive House")
                    bpm_in = gr.Number(label="BPM", value=130)
                    gen_btn = gr.Button("🚀 Test AI Generation (ZeroGPU)", variant="primary")

                with gr.Column(scale=1):
                    output_code = gr.Code(label="Generated Strudel Code (JavaScript)", language="javascript", lines=16)

            gen_btn.click(
                fn=generate_strudel_code,
                inputs=[prompt_in, code_in, genre_in, bpm_in],
                outputs=output_code,
                api_name="generate_pattern"
            )

        with gr.TabItem("📚 Dataset Curator & YouTube Scraping"):
            gr.Markdown("### Add Curated Strudel Code Patterns to the AI Training Dataset")
            with gr.Row():
                with gr.Column():
                    cur_prompt = gr.Textbox(label="Instruction / Musical Description", placeholder="e.g. Euclidean 5/8 drum groove with TR-909 bank")
                    cur_code = gr.Code(label="Valid Strudel JavaScript Code", language="javascript", lines=8)
                    cur_genre = gr.Dropdown(label="Genre", choices=["Progressive House", "Techno", "Trance", "Drum & Bass", "Ambient"], value="Progressive House")
                    cur_bpm = gr.Number(label="BPM", value=130)
                    add_btn = gr.Button("💾 Save Sample to Training Dataset", variant="primary")
                    cur_status = gr.Textbox(label="Status / Dataset Size", value=f"Total dataset entries: {get_dataset_count()}")

            add_btn.click(
                fn=add_to_dataset,
                inputs=[cur_prompt, cur_code, cur_genre, cur_bpm],
                outputs=cur_status
            )

        with gr.TabItem("🔥 ZeroGPU LoRA Fine-Tuning Studio"):
            gr.Markdown("### Fine-tune Qwen 2.5 Coder 7B on your Curated Dataset using ZeroGPU A100")
            with gr.Row():
                with gr.Column():
                    epoch_in = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Epochs")
                    batch_in = gr.Slider(minimum=1, maximum=8, value=2, step=1, label="Batch Size")
                    lr_in = gr.Textbox(label="Learning Rate", value="0.0002")
                    rank_in = gr.Dropdown(label="LoRA Rank (r)", choices=["8", "16", "32"], value="16")
                    train_btn = gr.Button("🔥 Start ZeroGPU Fine-Tuning Job", variant="primary")
                    train_status = gr.Textbox(label="Training Output & Logs", lines=6)

            train_btn.click(
                fn=run_lora_finetuning,
                inputs=[epoch_in, batch_in, lr_in, rank_in],
                outputs=train_status
            )

demo.queue().launch()
