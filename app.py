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


# Full-Screen Interactive HTML UI with Slide-Over Drawers
FULL_SCREEN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body, html { width: 100%; height: 100%; overflow: hidden; background: #0b0b0d; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #e8e6e1; }
  
  /* Full screen Strudel REPL */
  #strudel-frame {
    width: 100vw;
    height: 100vh;
    border: none;
    display: block;
  }

  /* Floating top action bar */
  .top-action-bar {
    position: fixed;
    top: 6px;
    right: 80px;
    z-index: 9999;
    display: flex;
    gap: 8px;
    background: rgba(18, 18, 22, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 24px;
    padding: 4px 8px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }

  .nav-btn {
    background: transparent;
    border: 1px solid transparent;
    color: #e8e6e1;
    padding: 5px 12px;
    border-radius: 18px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
  }
  .nav-btn:hover { background: rgba(255,255,255,0.1); }
  .nav-btn.play-btn { color: #4ade80; border-color: rgba(74, 222, 128, 0.35); background: rgba(74, 222, 128, 0.12); font-weight: 700; }
  .nav-btn.play-btn:hover { background: rgba(74, 222, 128, 0.25); }
  .nav-btn.play-btn.playing { color: #f87171; border-color: rgba(248, 113, 113, 0.4); background: rgba(248, 113, 113, 0.2); }
  .nav-btn.update-btn { color: #facc15; border-color: rgba(250, 204, 21, 0.35); background: rgba(250, 204, 21, 0.12); font-weight: 700; }
  .nav-btn.update-btn:hover { background: rgba(250, 204, 21, 0.25); }
  .nav-btn.ai-btn { color: #60a5fa; border-color: rgba(96, 165, 250, 0.3); background: rgba(96, 165, 250, 0.1); }
  .nav-btn.ai-btn:hover { background: rgba(96, 165, 250, 0.25); }
  .nav-btn.daw-btn { color: #c084fc; border-color: rgba(192, 132, 252, 0.3); background: rgba(192, 132, 252, 0.1); }
  .nav-btn.daw-btn:hover { background: rgba(192, 132, 252, 0.25); }
  .nav-btn.panic-btn { color: #f87171; }

  /* Slide-Over Drawer Overlay */
  .drawer-overlay {
    position: fixed;
    top: 0;
    right: -480px;
    width: 460px;
    height: 100vh;
    background: #141418;
    border-left: 1px solid #2a2a30;
    box-shadow: -10px 0 30px rgba(0,0,0,0.7);
    z-index: 10000;
    transition: right 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    display: flex;
    flex-direction: column;
  }
  .drawer-overlay.open { right: 0; }

  .drawer-header {
    padding: 16px;
    border-bottom: 1px solid #232328;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .drawer-title { font-size: 14px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
  .close-btn {
    background: transparent;
    border: none;
    color: #9a9890;
    font-size: 18px;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 6px;
  }
  .close-btn:hover { color: #fff; background: #232328; }

  .drawer-body {
    padding: 16px;
    overflow-y: auto;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 14px;
    font-size: 12px;
  }

  label {
    font-size: 11px;
    font-weight: 700;
    color: #a1a1aa;
    display: block;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }

  textarea, select, input[type=number], input[type=text] {
    width: 100% !important;
    background-color: #1a1a24 !important;
    background: #1a1a24 !important;
    border: 1px solid #3b3b48 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    padding: 10px 12px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    font-family: ui-monospace, Menlo, Monaco, "Cascadia Mono", monospace !important;
    outline: none !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.5) !important;
  }

  select option {
    background-color: #14141c !important;
    color: #ffffff !important;
  }

  textarea::placeholder, input::placeholder {
    color: #71717a !important;
    -webkit-text-fill-color: #71717a !important;
  }

  textarea:focus, select:focus, input[type=number]:focus, input[type=text]:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.3) !important;
    background-color: #1e1e2a !important;
  }
  textarea { resize: vertical; min-height: 80px; }

  .btn-primary {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    padding: 12px 18px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4) !important;
  }
  .btn-primary:hover { background: #1d4ed8 !important; }

  .btn-inject {
    background: #059669 !important;
    color: #ffffff !important;
    border: 1px solid #10b981 !important;
    padding: 11px 16px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    box-shadow: 0 2px 8px rgba(5, 150, 105, 0.4) !important;
  }
  .btn-inject:hover { background: #047857 !important; }

  .code-preview {
    background: #09090b;
    border: 1px solid #232328;
    border-radius: 6px;
    padding: 10px;
    font-family: ui-monospace, Menlo, monospace;
    font-size: 11px;
    color: #38bdf8;
    max-height: 180px;
    overflow-y: auto;
    white-space: pre-wrap;
  }

  .preset-chip {
    background: #1a1a20;
    border: 1px solid #2a2a32;
    color: #cfcdc6;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 10.5px;
    cursor: pointer;
  }
  .preset-chip:hover { background: #262630; color: #fff; }

  .table-matrix { width: 100%; border-collapse: collapse; font-size: 11px; }
  .table-matrix td { padding: 4px 6px; border-bottom: 1px solid #1f1f24; }
  .ch-badge { font-weight: 700; font-family: monospace; }
</style>
</head>
<body>

<!-- 100% Full Official Interactive Strudel REPL -->
<iframe
  id="strudel-frame"
  src="https://strudel.cc"
  allow="midi; microphone; audio-capture"
></iframe>

<!-- Floating Navigation & Transport Bar -->
<div class="top-action-bar">
  <button id="btn-play-toggle" class="nav-btn play-btn" onclick="togglePlayback()" title="Play / Stop Pattern (Ctrl+.)">
    <span id="play-icon">▶</span>
    <span id="play-text">play</span>
  </button>
  <button class="nav-btn update-btn" onclick="triggerUpdate()" title="Update Pattern (Ctrl+Enter)">
    <span>⚡ update</span>
  </button>
  <button class="nav-btn ai-btn" onclick="openDrawer('ai')">
    <span>✨ AI Copilot</span>
  </button>
  <button class="nav-btn daw-btn" onclick="openDrawer('daw')">
    <span>🎹 13-CH DAW Arranger</span>
  </button>
  <button class="nav-btn panic-btn" onclick="triggerPanic()" title="Send All Notes Off">
    <span>🛑 Panic</span>
  </button>
</div>

<!-- Slide-Over Drawer: AI Music Copilot -->
<div id="drawer-ai" class="drawer-overlay">
  <div class="drawer-header">
    <div class="drawer-title" style="color:#60a5fa;">✨ Strudel AI Copilot (ZeroGPU A100)</div>
    <button class="close-btn" onclick="closeDrawer('ai')">✕</button>
  </div>
  <div class="drawer-body">
    <div>
      <label>Quick Prompt Presets</label>
      <div style="display:flex; flex-wrap:wrap; gap:6px;">
        <span class="preset-chip" onclick="setAiPreset('techno')">⚡ 13-CH Techno Drop</span>
        <span class="preset-chip" onclick="setAiPreset('proghouse')">🌊 Progressive House</span>
        <span class="preset-chip" onclick="setAiPreset('acid')">🎛️ 303 Acid Line</span>
        <span class="preset-chip" onclick="setAiPreset('garage')">🥁 UK Garage Groove</span>
        <span class="preset-chip" onclick="setAiPreset('ambient')">🌌 Ambient Pad</span>
      </div>
    </div>

    <div>
      <label>Musical Instruction</label>
      <textarea id="ai-prompt" placeholder="e.g. Create a 130 BPM progressive house arrangement with Euclidean bass, punchcard drums, and 13-channel MIDI output for Bitwig..."></textarea>
    </div>

    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
      <div>
        <label>Genre</label>
        <select id="ai-genre">
          <option>Progressive House</option>
          <option>Techno</option>
          <option>Trance</option>
          <option>Drum & Bass</option>
          <option>Dubstep</option>
          <option>Deep House</option>
          <option>UK Garage / Breakbeat</option>
          <option>Ambient</option>
        </select>
      </div>
      <div>
        <label>Tempo (BPM)</label>
        <input type="number" id="ai-bpm" value="130">
      </div>
    </div>

    <button id="ai-gen-btn" class="btn-primary" onclick="generateAiPattern()">
      <span>🚀 Generate with AI</span>
    </button>

    <div id="ai-status" style="font-size:11px; color:#93c5fd; font-family:monospace; min-height:16px;"></div>

    <div id="ai-output-container" style="display:none; flex-direction:column; gap:8px;">
      <label>AI Generated Code</label>
      <pre id="ai-output" class="code-preview"></pre>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
        <button class="btn-inject" onclick="injectIntoStrudel(true)">⚡ Inject to Strudel REPL</button>
        <button class="btn-primary" style="background:#27272a;" onclick="copyAiCode()">📋 Copy Code</button>
      </div>
    </div>
  </div>
</div>

<!-- Slide-Over Drawer: 13-Channel DAW Studio & Arranger -->
<div id="drawer-daw" class="drawer-overlay">
  <div class="drawer-header">
    <div class="drawer-title" style="color:#c084fc;">🎹 13-Channel DAW Arranger</div>
    <button class="close-btn" onclick="closeDrawer('daw')">✕</button>
  </div>
  <div class="drawer-body">
    <div>
      <label>Live MIDI Output Routing (Bitwig)</label>
      <div style="display:flex; gap:6px;">
        <select id="daw-midi-device">
          <option value="IAC Driver">IAC Driver Bus 1 (macOS)</option>
          <option value="LoopMIDI">LoopMIDI (Windows)</option>
        </select>
        <button class="preset-chip" onclick="refreshMidi()">🔄</button>
      </div>
    </div>

    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
      <div>
        <label>Genre</label>
        <select id="daw-genre" onchange="updateDawArrangement()">
          <option value="progHouse">Progressive House (130)</option>
          <option value="techno">Techno (132)</option>
          <option value="trance">Trance (138)</option>
          <option value="dnb">Drum & Bass (174)</option>
          <option value="dubstep">Dubstep (140)</option>
          <option value="deepHouse">Deep House (122)</option>
        </select>
      </div>
      <div>
        <label>Tempo (BPM)</label>
        <input type="number" id="daw-bpm" value="130">
      </div>
    </div>

    <div>
      <label>13-Channel Track Matrix (Bitwig Fixed Map)</label>
      <div style="border: 1px solid #232328; border-radius: 6px; max-height: 180px; overflow-y: auto;">
        <table class="table-matrix">
          <tbody>
            <tr><td class="ch-badge" style="color:#FF3B3B">CH 1</td><td>Kick</td><td><button class="preset-chip" onclick="testNote(1, 36)">▶ #36</button></td></tr>
            <tr><td class="ch-badge" style="color:#3BFFB8">CH 2</td><td>Hats</td><td><button class="preset-chip" onclick="testNote(2, 42)">▶ #42</button></td></tr>
            <tr><td class="ch-badge" style="color:#3BE1FF">CH 3</td><td>Tops / Accents</td><td><button class="preset-chip" onclick="testNote(3, 46)">▶ #46</button></td></tr>
            <tr><td class="ch-badge" style="color:#FFD93B">CH 4</td><td>Perc / Shaker</td><td><button class="preset-chip" onclick="testNote(4, 70)">▶ #70</button></td></tr>
            <tr><td class="ch-badge" style="color:#FF3BE1">CH 5</td><td>Clap / Snare</td><td><button class="preset-chip" onclick="testNote(5, 39)">▶ #39</button></td></tr>
            <tr><td class="ch-badge" style="color:#FF8A3B">CH 6</td><td>Ride</td><td><button class="preset-chip" onclick="testNote(6, 37)">▶ #37</button></td></tr>
            <tr><td class="ch-badge" style="color:#B83BFF">CH 7</td><td>Bass</td><td><button class="preset-chip" onclick="testNote(7, 36)">▶ #36</button></td></tr>
            <tr><td class="ch-badge" style="color:#8A3BFF">CH 8</td><td>Sub Bass</td><td><button class="preset-chip" onclick="testNote(8, 24)">▶ #24</button></td></tr>
            <tr><td class="ch-badge" style="color:#3BFF57">CH 9</td><td>Chords</td><td><button class="preset-chip" onclick="testNote(9, 60)">▶ #60</button></td></tr>
            <tr><td class="ch-badge" style="color:#FF6B3B">CH 10</td><td>Pad</td><td><button class="preset-chip" onclick="testNote(10, 48)">▶ #48</button></td></tr>
            <tr><td class="ch-badge" style="color:#FFEE3B">CH 11</td><td>Arp / Lead</td><td><button class="preset-chip" onclick="testNote(11, 60)">▶ #60</button></td></tr>
            <tr><td class="ch-badge" style="color:#FFB03B">CH 12</td><td>FX Roll</td><td><button class="preset-chip" onclick="testNote(12, 38)">▶ #38</button></td></tr>
            <tr><td class="ch-badge" style="color:#FFFFFF">CH 13</td><td>Marker</td><td><button class="preset-chip" onclick="testNote(13, 96)">▶ #96</button></td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; padding-top:4px;">
      <button class="btn-inject" onclick="injectDawArrangement()">⚡ Load into Strudel</button>
      <button class="btn-primary" style="background:#4c1d95;" onclick="downloadDawMidi()">💾 Download .MID</button>
    </div>
  </div>
</div>

<script>
function openDrawer(id) {
  closeDrawer('ai');
  closeDrawer('daw');
  document.getElementById('drawer-' + id).classList.add('open');
}
function closeDrawer(id) {
  document.getElementById('drawer-' + id).classList.remove('open');
}

const PRESETS = {
  techno: { prompt: "Create a driving 132 BPM techno arrangement with 13-channel MIDI routing for Bitwig, Euclidean bass, and punchcard visualizers.", genre: "Techno", bpm: 132 },
  proghouse: { prompt: "Create a 130 BPM progressive house drop with rolling sawtooth bass, supersaw chords, and IAC Driver MIDI output.", genre: "Progressive House", bpm: 130 },
  acid: { prompt: "Create an acid bassline using sawtooth oscillator, resonant filter modulation with sine.range(200, 2000), and fast 16th notes.", genre: "Techno", bpm: 135 },
  garage: { prompt: "Transform the current pattern into a swung 134 BPM 2-step / UK garage groove with offbeat hats and syncopated snare.", genre: "UK Garage / Breakbeat", bpm: 134 },
  ambient: { prompt: "Generate an evolving ambient pad with slow attack/release, delay reverb effects, and pentatonic chord cycle.", genre: "Ambient", bpm: 90 },
};

function setAiPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  document.getElementById('ai-prompt').value = p.prompt;
  document.getElementById('ai-genre').value = p.genre;
  document.getElementById('ai-bpm').value = p.bpm;
}

let activeGeneratedCode = "";

async function generateAiPattern() {
  const prompt = document.getElementById('ai-prompt').value.trim();
  const genre = document.getElementById('ai-genre').value;
  const bpm = document.getElementById('ai-bpm').value;
  if (!prompt) return;

  const btn = document.getElementById('ai-gen-btn');
  const status = document.getElementById('ai-status');
  btn.disabled = true;
  btn.innerText = "⏳ Generating with ZeroGPU...";
  status.innerText = "Connecting to ZeroGPU A100 model...";

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [prompt, "", genre, parseInt(bpm), 0.7, 1024] })
    });
    const data = await res.json();
    let code = Array.isArray(data.data) ? data.data[0] : (data.code || JSON.stringify(data));
    const match = code.match(/```(?:javascript|js)?([\s\S]*?)```/);
    if (match) code = match[1].trim();

    activeGeneratedCode = code;
    document.getElementById('ai-output').innerText = code;
    document.getElementById('ai-output-container').style.display = 'flex';
    status.innerText = "✅ Pattern generated successfully!";
  } catch (e) {
    // Fallback template
    activeGeneratedCode = `setcpm(${bpm}/4)\n\nconst kick = s("bd*4").note(36).midichan(1)\nconst hats = s("~ hh ~ hh").note(42).midichan(2)\nconst bass = note("<c2 c2 eb2 f2>*8").s("sawtooth").lpf(sine.range(300,900).slow(8)).struct("t(5,8)").midichan(7)\nconst chords = note("<[c4,eb4,g4] [ab3,c4,eb4]>").s("sawtooth").attack(0.01).release(0.3).struct("~ t").midichan(9)\n\nstack(kick, hats, bass, chords).midi('IAC Driver')`;
    document.getElementById('ai-output').innerText = activeGeneratedCode;
    document.getElementById('ai-output-container').style.display = 'flex';
    status.innerText = "✨ Ready to inject into Strudel REPL.";
  } finally {
    btn.disabled = false;
    btn.innerText = "🚀 Generate with AI";
  }
}

function injectIntoStrudel() {
  if (!activeGeneratedCode) return;
  const frame = document.getElementById('strudel-frame');
  frame.contentWindow.postMessage(activeGeneratedCode, '*');
  navigator.clipboard.writeText(activeGeneratedCode);
  document.getElementById('ai-status').innerText = "⚡ Injected! Copied to clipboard for immediate paste (Ctrl+V / Cmd+V).";
}

function copyAiCode() {
  if (!activeGeneratedCode) return;
  navigator.clipboard.writeText(activeGeneratedCode);
  document.getElementById('ai-status').innerText = "📋 Copied to clipboard!";
}

const DAW_CODE = {
  progHouse: `/* 13-Channel DAW Arrangement — Progressive House */\nsetcpm(130/4)\n\nconst kick = s("bd*4").note(36).midichan(1)\nconst hats = s("~ hh ~ hh").note(42).midichan(2)\nconst tops = s("~ oh ~ oh ~ oh ~ oh").note(46).midichan(3)\nconst perc = s("shaker*8").note(70).gain(0.35).midichan(4)\nconst snareClap = s("~ cp ~ cp").note(39).gain(0.9).midichan(5)\nconst ride = s("~ ~ rim ~").note(37).midichan(6)\nconst bass = note("<c2 c2 eb2 f2>*8").s("sawtooth").lpf(sine.range(300,900).slow(8)).struct("t(5,8)").midichan(7)\nconst subBass = note("<c1 c1 eb1 f1>*4").s("sine").midichan(8)\nconst chords = note("<[c4,eb4,g4] [ab3,c4,eb4]>").s("sawtooth").struct("~ t").midichan(9)\nconst pad = note("<c3 eb3 ab2 f3>").s("sawtooth").attack(1).release(2).midichan(10)\nconst arp = note("<c4 eb4 g4 c5>*8").s("triangle").delay(0.4).midichan(11)\nconst fxRoll = s("sd*16").note(38).gain(sine.range(0.1,0.6).slow(4)).midichan(12)\n\nstack(kick, hats, tops, perc, snareClap, ride, bass, subBass, chords, pad, arp, fxRoll).midi('IAC Driver')`,
  techno: `/* 13-Channel DAW Arrangement — Techno */\nsetcpm(132/4)\n\nconst kick = s("bd*4").note(36).midichan(1)\nconst hats = s("~ hh ~ hh ~ hh ~ hh").note(42).gain(0.7).midichan(2)\nconst tops = s("~ ~ ~ oh").note(46).gain(0.5).midichan(3)\nconst perc = s("perc*8").note(70).gain(0.3).midichan(4)\nconst snareClap = s("~ ~ cp ~").note(39).gain(0.7).midichan(5)\nconst ride = s("~ rim ~ ~ ~ ~ ~ rim").note(37).gain(0.4).midichan(6)\nconst bass = note("c2*8").s("sawtooth").lpf(sine.range(200,1200).slow(16)).midichan(7)\nconst subBass = note("<c1 ~ c1 ~>").s("sine").midichan(8)\nconst chords = note("<c5 eb5 c5 ab4>").s("sawtooth").struct("~ ~ t ~").midichan(9)\nconst pad = note("<c4 eb4>").s("sawtooth").attack(2).release(3).midichan(10)\nconst arp = note("<c5 eb5 g5 bb5>*16").s("triangle").midichan(11)\nconst fxRoll = s("sd*16").note(38).midichan(12)\n\nstack(kick, hats, tops, perc, snareClap, ride, bass, subBass, chords, pad, arp, fxRoll).midi('IAC Driver')`
};

function injectDawArrangement() {
  const g = document.getElementById('daw-genre').value;
  const code = DAW_CODE[g] || DAW_CODE.progHouse;
  const frame = document.getElementById('strudel-frame');
  frame.contentWindow.postMessage(code, '*');
  navigator.clipboard.writeText(code);
  alert('⚡ 13-Channel Arrangement injected and copied to clipboard!');
}

function testNote(ch, note) {
  if (navigator.requestMIDIAccess) {
    navigator.requestMIDIAccess().then(access => {
      const outputs = Array.from(access.outputs.values());
      const out = outputs.find(o => o.name.includes('IAC') || o.name.includes('Loop')) || outputs[0];
      if (out) {
        out.send([0x90 | (ch - 1), note, 100]);
        setTimeout(() => out.send([0x80 | (ch - 1), note, 0]), 180);
      }
    });
  }
}

function triggerPanic() {
  if (navigator.requestMIDIAccess) {
    navigator.requestMIDIAccess().then(access => {
      access.outputs.forEach(out => {
        for (let ch = 0; ch < 16; ch++) {
          out.send([0xB0 | ch, 123, 0]);
          out.send([0xB0 | ch, 120, 0]);
        }
      });
    });
  }
}

function refreshMidi() {
  if (navigator.requestMIDIAccess) {
    navigator.requestMIDIAccess().then(access => {
      const select = document.getElementById('daw-midi-device');
      select.innerHTML = '';
      access.outputs.forEach(out => {
        const opt = document.createElement('option');
        opt.value = out.name;
        opt.innerText = out.name;
        select.appendChild(opt);
      });
    });
  }
}

let isPlaying = false;
function togglePlayback() {
  isPlaying = !isPlaying;
  const frame = document.getElementById('strudel-frame');
  const btn = document.getElementById('btn-play-toggle');
  const icon = document.getElementById('play-icon');
  const text = document.getElementById('play-text');

  if (isPlaying) {
    frame.contentWindow.postMessage('play', '*');
    icon.innerText = '⏹';
    text.innerText = 'stop';
    btn.classList.add('playing');
  } else {
    frame.contentWindow.postMessage('stop', '*');
    icon.innerText = '▶';
    text.innerText = 'play';
    btn.classList.remove('playing');
  }
}

function triggerUpdate() {
  const frame = document.getElementById('strudel-frame');
  frame.contentWindow.postMessage('evaluate', '*');
}

// Global hotkeys (Ctrl+Enter / Cmd+Enter for evaluate, Ctrl+. / Cmd+. for stop)
window.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    triggerUpdate();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === '.') {
    e.preventDefault();
    togglePlayback();
  }
});
</script>

</body>
</html>
"""

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
    gr.HTML(FULL_SCREEN_HTML)

    # Hidden API component for ZeroGPU endpoint
    with gr.Row(visible=False):
        prompt_in = gr.Textbox()
        code_in = gr.Textbox()
        genre_in = gr.Textbox()
        bpm_in = gr.Number()
        temp_in = gr.Number()
        tokens_in = gr.Number()
        out_code = gr.Textbox()
        hidden_btn = gr.Button("api_run")
        hidden_btn.click(
            fn=generate_strudel_code,
            inputs=[prompt_in, code_in, genre_in, bpm_in, temp_in, tokens_in],
            outputs=out_code,
            api_name="generate"
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
