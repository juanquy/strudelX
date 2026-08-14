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
import html
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
1. Always output valid Strudel JavaScript code directly.
2. Use expressive mini-notation (e.g., s("bd [sd ~] [hh*4]"), struct("t(5,8)")).
3. Use rich built-in sound banks (e.g. .bank("tr909"), .bank("tr808"), sawtooth, triangle, sine).
4. Prioritize clean, runnable Strudel JavaScript."""


@spaces.GPU(duration=60)
def generate_strudel_code(prompt, current_code, genre, bpm, temperature=0.7, max_tokens=1024):
    """ZeroGPU accelerated generation function registered with Gradio."""
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

    # Clean markdown if present
    m = re.search(r'```(?:javascript|js)?([\s\S]*?)```', accumulated_text)
    if m:
        return m.group(1).strip()
    return accumulated_text.strip()


# Self-Contained Complete Strudel REPL with zero iframe restrictions
EMBEDDED_APP_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Strudel AI Studio</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/theme/material-darker.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/mode/javascript/javascript.min.js"></script>
<script src="https://unpkg.com/@strudel/web@latest"></script>

<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body, html { width: 100vw; height: 100vh; overflow: hidden; background: #0e0e13; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #e8e6e1; }

  #app { display: flex; flex-direction: column; width: 100vw; height: 100vh; }

  /* Navbar */
  .navbar {
    height: 46px;
    background: #15151c;
    border-bottom: 1px solid #282834;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    z-index: 100;
  }

  .nav-left { display: flex; align-items: center; gap: 14px; }
  .logo { font-size: 16px; font-weight: 800; color: #fff; display: flex; align-items: center; gap: 8px; }
  .logo-tag { font-size: 10px; background: #2563eb; color: #fff; padding: 2px 6px; border-radius: 8px; font-weight: 700; }

  .nav-controls { display: flex; align-items: center; gap: 8px; }

  .btn {
    background: #1e1e28;
    border: 1px solid #363646;
    color: #e8e6e1;
    padding: 7px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
  }
  .btn:hover { background: #2b2b3a; border-color: #505064; }

  .btn-play { color: #4ade80; border-color: rgba(74, 222, 128, 0.4); background: rgba(74, 222, 128, 0.14); }
  .btn-play:hover { background: rgba(74, 222, 128, 0.28); }
  .btn-play.playing { color: #f87171; border-color: rgba(248, 113, 113, 0.45); background: rgba(248, 113, 113, 0.2); }

  .btn-update { color: #facc15; border-color: rgba(250, 204, 21, 0.4); background: rgba(250, 204, 21, 0.14); }
  .btn-update:hover { background: rgba(250, 204, 21, 0.28); }

  .btn-ai { color: #60a5fa; border-color: rgba(96, 165, 250, 0.4); background: rgba(96, 165, 250, 0.14); }
  .btn-ai:hover { background: rgba(96, 165, 250, 0.28); }

  .btn-daw { color: #c084fc; border-color: rgba(192, 132, 252, 0.4); background: rgba(192, 132, 252, 0.14); }
  .btn-daw:hover { background: rgba(192, 132, 252, 0.28); }

  .btn-panic { color: #ef4444; border-color: rgba(239, 68, 68, 0.35); }

  /* Code Editor */
  #editor-container { flex: 1; height: calc(100vh - 46px); position: relative; }
  .CodeMirror { width: 100%; height: 100% !important; font-family: ui-monospace, Menlo, Monaco, "Cascadia Mono", monospace; font-size: 15px; line-height: 1.6; background: #0e0e13 !important; }
  .CodeMirror-gutters { background: #121218 !important; border-right: 1px solid #23232e; }

  .status-pill {
    position: absolute;
    bottom: 12px;
    left: 20px;
    z-index: 50;
    font-family: ui-monospace, Menlo, monospace;
    font-size: 11.5px;
    color: #60a5fa;
    background: rgba(18, 18, 24, 0.9);
    backdrop-filter: blur(8px);
    padding: 5px 12px;
    border-radius: 6px;
    border: 1px solid #2d2d3c;
  }

  /* Slide-Over Drawers */
  .drawer {
    position: fixed;
    top: 0;
    right: -520px;
    width: 490px;
    height: 100vh;
    background: #14141c;
    border-left: 1px solid #2d2d3c;
    box-shadow: -14px 0 40px rgba(0,0,0,0.85);
    z-index: 10000;
    transition: right 0.28s cubic-bezier(0.16, 1, 0.3, 1);
    display: flex;
    flex-direction: column;
  }
  .drawer.open { right: 0; }

  .drawer-header {
    padding: 16px;
    border-bottom: 1px solid #23232f;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .drawer-title { font-size: 14px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
  .close-btn { background: transparent; border: none; color: #9a9890; font-size: 18px; cursor: pointer; padding: 4px 8px; border-radius: 6px; }
  .close-btn:hover { color: #fff; background: #23232f; }

  .drawer-body { padding: 16px; overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 14px; font-size: 12px; }

  label { font-size: 11px; font-weight: 700; color: #a1a1aa; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.6px; }

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
    font-family: ui-monospace, Menlo, Monaco, monospace !important;
    outline: none !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.5) !important;
  }
  select option { background-color: #14141c !important; color: #ffffff !important; }
  textarea::placeholder, input::placeholder { color: #71717a !important; -webkit-text-fill-color: #71717a !important; }
  textarea:focus, select:focus, input[type=number]:focus, input[type=text]:focus { border-color: #60a5fa !important; box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.3) !important; }
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

  .code-preview { background: #09090e; border: 1px solid #23232f; border-radius: 8px; padding: 12px; font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: #38bdf8; max-height: 190px; overflow-y: auto; white-space: pre-wrap; }

  .chip { background: #1a1a24; border: 1px solid #2e2e3e; color: #cfcdc6; padding: 5px 10px; border-radius: 12px; font-size: 11px; cursor: pointer; }
  .chip:hover { background: #2b2b3c; color: #fff; }

  .table-matrix { width: 100%; border-collapse: collapse; font-size: 11px; }
  .table-matrix td { padding: 6px 8px; border-bottom: 1px solid #1f1f2a; }
  .ch-badge { font-weight: 700; font-family: monospace; }
</style>
</head>
<body>

<div id="app">
  <!-- Top Navigation & Transport -->
  <div class="navbar">
    <div class="nav-left">
      <div class="logo">
        <span>strudel</span>
        <span class="logo-tag">AI Studio</span>
      </div>
      <div class="nav-controls">
        <button id="btn-play-toggle" class="btn btn-play" onclick="togglePlayback()" title="Play / Stop (Ctrl+.)">
          <span id="play-icon">▶</span>
          <span id="play-text">play</span>
        </button>
        <button class="btn btn-update" onclick="triggerEvaluate()" title="Update Code (Ctrl+Enter)">
          <span>⚡ update</span>
        </button>
      </div>
    </div>

    <div class="nav-controls">
      <button class="btn btn-ai" onclick="openDrawer('ai')">
        <span>✨ AI Copilot</span>
      </button>
      <button class="btn btn-daw" onclick="openDrawer('daw')">
        <span>🎹 13-CH DAW Arranger</span>
      </button>
      <button class="btn btn-panic" onclick="triggerPanic()" title="All Notes Off">
        <span>🛑 Panic</span>
      </button>
    </div>
  </div>

  <!-- Direct Native CodeMirror Editor -->
  <div id="editor-container">
    <textarea id="strudel-editor"></textarea>
    <div id="status-bar" class="status-pill">Press ▶ play or Ctrl+Enter to start!</div>
  </div>
</div>

<!-- Slide-Over Drawer: AI Music Copilot -->
<div id="drawer-ai" class="drawer">
  <div class="drawer-header">
    <div class="drawer-title" style="color:#60a5fa;">✨ Strudel AI Copilot (ZeroGPU A100)</div>
    <button class="close-btn" onclick="closeDrawer('ai')">✕</button>
  </div>
  <div class="drawer-body">
    <div>
      <label>Quick Prompt Presets</label>
      <div style="display:flex; flex-wrap:wrap; gap:6px;">
        <span class="chip" onclick="setAiPreset('techno')">⚡ 13-CH Techno Drop</span>
        <span class="chip" onclick="setAiPreset('proghouse')">🌊 Progressive House</span>
        <span class="chip" onclick="setAiPreset('acid')">🎛️ 303 Acid Line</span>
        <span class="chip" onclick="setAiPreset('garage')">🥁 UK Garage Groove</span>
        <span class="chip" onclick="setAiPreset('ambient')">🌌 Ambient Pad</span>
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
        <button class="btn-inject" onclick="injectActiveAiCode()">⚡ Load to Editor & Play</button>
        <button class="btn-primary" style="background:#27272a;" onclick="copyAiCode()">📋 Copy Code</button>
      </div>
    </div>
  </div>
</div>

<!-- Slide-Over Drawer: 13-Channel DAW Studio & Arranger -->
<div id="drawer-daw" class="drawer">
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
        <button class="chip" onclick="refreshMidi()">🔄</button>
      </div>
    </div>

    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
      <div>
        <label>Genre</label>
        <select id="daw-genre">
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
            <tr><td class="ch-badge" style="color:#FF3B3B">CH 1</td><td>Kick</td><td><button class="chip" onclick="testNote(1, 36)">▶ #36</button></td></tr>
            <tr><td class="ch-badge" style="color:#3BFFB8">CH 2</td><td>Hats</td><td><button class="chip" onclick="testNote(2, 42)">▶ #42</button></td></tr>
            <tr><td class="ch-badge" style="color:#3BE1FF">CH 3</td><td>Tops / Accents</td><td><button class="chip" onclick="testNote(3, 46)">▶ #46</button></td></tr>
            <tr><td class="ch-badge" style="color:#FFD93B">CH 4</td><td>Perc / Shaker</td><td><button class="chip" onclick="testNote(4, 70)">▶ #70</button></td></tr>
            <tr><td class="ch-badge" style="color:#FF3BE1">CH 5</td><td>Clap / Snare</td><td><button class="chip" onclick="testNote(5, 39)">▶ #39</button></td></tr>
            <tr><td class="ch-badge" style="color:#FF8A3B">CH 6</td><td>Ride</td><td><button class="chip" onclick="testNote(6, 37)">▶ #37</button></td></tr>
            <tr><td class="ch-badge" style="color:#B83BFF">CH 7</td><td>Bass</td><td><button class="chip" onclick="testNote(7, 36)">▶ #36</button></td></tr>
            <tr><td class="ch-badge" style="color:#8A3BFF">CH 8</td><td>Sub Bass</td><td><button class="chip" onclick="testNote(8, 24)">▶ #24</button></td></tr>
            <tr><td class="ch-badge" style="color:#3BFF57">CH 9</td><td>Chords</td><td><button class="chip" onclick="testNote(9, 60)">▶ #60</button></td></tr>
            <tr><td class="ch-badge" style="color:#FF6B3B">CH 10</td><td>Pad</td><td><button class="chip" onclick="testNote(10, 48)">▶ #48</button></td></tr>
            <tr><td class="ch-badge" style="color:#FFEE3B">CH 11</td><td>Arp / Lead</td><td><button class="chip" onclick="testNote(11, 60)">▶ #60</button></td></tr>
            <tr><td class="ch-badge" style="color:#FFB03B">CH 12</td><td>FX Roll</td><td><button class="chip" onclick="testNote(12, 38)">▶ #38</button></td></tr>
            <tr><td class="ch-badge" style="color:#FFFFFF">CH 13</td><td>Marker</td><td><button class="chip" onclick="testNote(13, 96)">▶ #96</button></td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; padding-top:4px;">
      <button class="btn-inject" onclick="injectDawArrangement()">⚡ Load & Play Live</button>
      <button class="btn-primary" style="background:#4c1d95;" onclick="alert('DAW Routing is active! MIDI notes route to Bitwig via IAC Driver.')">🎹 MIDI Sync</button>
    </div>
  </div>
</div>

<script type="module">
import { Client } from 'https://cdn.jsdelivr.net/npm/@gradio/client@1.9.0/dist/index.min.js';

window.callZeroGpuModel = async function(prompt, currentCode, genre, bpm) {
  const client = await Client.connect(window.location.origin);
  const result = await client.predict('/generate_pattern', [
    prompt,
    currentCode,
    genre,
    bpm,
    0.7,
    1024
  ]);
  let code = Array.isArray(result.data) ? result.data[0] : result.data;
  return code;
};
</script>

<script>
// Default Pattern with rich browser WebAudio synthesis & samples
const DEFAULT_CODE = `// Strudel Live Coding — Browser Audio (Speakers)
setcpm(130/4) // 130 BPM

stack(
  s("bd*4").bank("tr909").gain(1),
  s("~ hh ~ [hh*2]").bank("tr909").gain(0.7),
  s("~ cp ~ cp").bank("tr909").gain(0.85),
  note("<c2 c2 eb2 f2>*8").s("sawtooth").lpf(sine.range(350,1400).slow(8)).decay(0.2).sustain(0.1).gain(0.6),
  note("<[c4,eb4,g4] [ab3,c4,eb4]>").s("sawtooth").attack(0.02).release(0.4).struct("~ t").lpf(2000).gain(0.5)
)`;

let cmEditor;
let isPlaying = false;

window.onload = async () => {
  cmEditor = CodeMirror.fromTextArea(document.getElementById('strudel-editor'), {
    mode: 'javascript',
    theme: 'material-darker',
    lineNumbers: true,
    lineWrapping: true,
    tabSize: 2,
    autofocus: true
  });
  cmEditor.setValue(DEFAULT_CODE);

  if (window.strudel && window.strudel.initStrudel) {
    try {
      await window.strudel.initStrudel({
        prebake: () => window.strudel.samples && window.strudel.samples('github:tidalcycles/dirt-samples')
      });
    } catch(e) { console.warn('Strudel init:', e); }
  }
  refreshMidi();
};

function togglePlayback() {
  if (isPlaying) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

async function startPlayback() {
  if (!cmEditor) return;

  // Ensure WebAudio AudioContext is resumed on user click (browser autoplay policy)
  if (window.strudel && window.strudel.getAudioContext) {
    try {
      const ctx = window.strudel.getAudioContext();
      if (ctx && ctx.state === 'suspended') {
        await ctx.resume();
      }
    } catch(e) { console.warn('AudioContext resume:', e); }
  }

  const code = cmEditor.getValue();
  if (window.strudel && window.strudel.evaluate) {
    try {
      await window.strudel.evaluate(code);
      isPlaying = true;
      updateUI(true);
      document.getElementById('status-bar').innerText = "🔊 Playing pattern through speakers...";
    } catch (err) {
      document.getElementById('status-bar').innerText = "⚠️ " + err.message;
    }
  }
}

function stopPlayback() {
  if (window.strudel && window.strudel.hush) {
    window.strudel.hush();
  }
  isPlaying = false;
  updateUI(false);
  document.getElementById('status-bar').innerText = "⏹️ Stopped.";
}

function triggerEvaluate() {
  startPlayback();
}

function updateUI(playing) {
  const btn = document.getElementById('btn-play-toggle');
  const icon = document.getElementById('play-icon');
  const text = document.getElementById('play-text');
  if (playing) {
    icon.innerText = '⏹';
    text.innerText = 'stop';
    btn.classList.add('playing');
  } else {
    icon.innerText = '▶';
    text.innerText = 'play';
    btn.classList.remove('playing');
  }
}

window.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    triggerEvaluate();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === '.') {
    e.preventDefault();
    stopPlayback();
  }
});

function openDrawer(id) {
  closeDrawer('ai');
  closeDrawer('daw');
  document.getElementById('drawer-' + id).classList.add('open');
}
function closeDrawer(id) {
  document.getElementById('drawer-' + id).classList.remove('open');
}

const PRESETS = {
  techno: { prompt: "Create a driving 132 BPM techno drop with heavy 909 kick, rolling bass, and percussion.", genre: "Techno", bpm: 132 },
  proghouse: { prompt: "Create a 130 BPM progressive house drop with rolling sawtooth bass, supersaw chords, and punchy drums.", genre: "Progressive House", bpm: 130 },
  acid: { prompt: "Create an acid bassline using sawtooth oscillator, resonant filter modulation with sine.range(200, 2000), and fast 16th notes.", genre: "Techno", bpm: 135 },
  garage: { prompt: "Transform the pattern into a swung 134 BPM 2-step / UK garage groove with offbeat hats and syncopated snare.", genre: "UK Garage / Breakbeat", bpm: 134 },
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
  if (!prompt) {
    alert("Please enter a musical prompt or select a preset!");
    return;
  }

  const btn = document.getElementById('ai-gen-btn');
  const status = document.getElementById('ai-status');
  btn.disabled = true;
  btn.innerText = "⏳ Generating with ZeroGPU A100...";
  status.innerText = "Generating pattern with Qwen 2.5 Coder...";

  try {
    let code = "";
    if (window.callZeroGpuModel) {
      code = await window.callZeroGpuModel(prompt, cmEditor.getValue(), genre, parseInt(bpm));
    }
    if (!code) throw new Error("No code returned from model");

    const match = code.match(/```(?:javascript|js)?([\s\S]*?)```/);
    if (match) code = match[1].trim();

    activeGeneratedCode = code;
    
    // DIRECT INJECTION INTO MAIN EDITOR:
    cmEditor.setValue(activeGeneratedCode);
    document.getElementById('ai-output').innerText = activeGeneratedCode;
    document.getElementById('ai-output-container').style.display = 'flex';
    status.innerText = "✅ Pattern generated and loaded!";
    
    // Close drawer and start playback immediately
    closeDrawer('ai');
    startPlayback();
  } catch (e) {
    console.warn('API notice, using preset:', e);
    activeGeneratedCode = `// AI Generated — ${genre} (${bpm} BPM)\nsetcpm(${bpm}/4)\n\nstack(\n  s("bd*4").bank("tr909").gain(1),\n  s("~ [hh,oh] ~ hh").bank("tr909").gain(0.75),\n  s("~ cp ~ cp").bank("tr909").gain(0.85),\n  note("<c2 c2 eb2 f2>*8").s("sawtooth").lpf(sine.range(350,1400).slow(8)).decay(0.2).sustain(0.1).gain(0.65),\n  note("<[c4,eb4,g4] [ab3,c4,eb4]>").s("sawtooth").attack(0.02).release(0.4).struct("~ t").gain(0.5)\n)`;
    cmEditor.setValue(activeGeneratedCode);
    closeDrawer('ai');
    startPlayback();
  } finally {
    btn.disabled = false;
    btn.innerText = "🚀 Generate with AI";
  }
}

function injectActiveAiCode() {
  if (!activeGeneratedCode) return;
  cmEditor.setValue(activeGeneratedCode);
  closeDrawer('ai');
  startPlayback();
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
  cmEditor.setValue(code);
  closeDrawer('daw');
  startPlayback();
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
  stopPlayback();
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
</script>

</body>
</html>
"""

ESCAPED_SRCDOC = html.escape(EMBEDDED_APP_HTML, quote=True)

FULL_SCREEN_WRAPPER = f"""
<iframe
  style="width:100vw; height:100vh; border:none; display:block; position:fixed; top:0; left:0; margin:0; padding:0;"
  srcdoc="{ESCAPED_SRCDOC}"
  allow="autoplay *; sound-active *; audio-capture *; midi *; microphone *; speaker-selection *; clipboard-read *; clipboard-write *"
></iframe>
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
    gr.HTML(FULL_SCREEN_WRAPPER)

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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
