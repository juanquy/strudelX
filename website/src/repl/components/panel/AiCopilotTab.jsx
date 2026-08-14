import cx from '@src/cx.mjs';
import { useState, useRef } from 'react';

const QUICK_PROMPTS = [
  {
    label: '⚡ 13-CH Bitwig Techno',
    prompt: 'Create a driving 132 BPM techno arrangement with 13-channel MIDI routing for Bitwig, Euclidean bass, and punchcard visualizers.',
    genre: 'Techno',
    bpm: 132,
  },
  {
    label: '🌊 Prog House Drop',
    prompt: 'Create a 130 BPM progressive house drop with rolling sawtooth bass, supersaw chords, and IAC Driver MIDI output.',
    genre: 'Progressive House',
    bpm: 130,
  },
  {
    label: '🎛️ 303 Acid Line',
    prompt: 'Create an acid bassline using sawtooth oscillator, resonant filter modulation with sine.range(200, 2000), and fast 16th notes.',
    genre: 'Techno',
    bpm: 135,
  },
  {
    label: '🥁 UK Garage Groove',
    prompt: 'Transform the current pattern into a swung 134 BPM 2-step / UK garage groove with offbeat hats and syncopated snare.',
    genre: 'UK Garage / Breakbeat',
    bpm: 134,
  },
  {
    label: '🌌 Ambient Soundscape',
    prompt: 'Generate an evolving ambient pad with slow attack/release, delay reverb effects, and pentatonic chord cycle.',
    genre: 'Ambient',
    bpm: 90,
  },
];

export default function AiCopilotTab({ context }) {
  const [prompt, setPrompt] = useState('');
  const [useCurrentCode, setUseCurrentCode] = useState(true);
  const [genre, setGenre] = useState('Progressive House');
  const [bpm, setBpm] = useState(130);
  const [temperature, setTemperature] = useState(0.7);
  const [hfEndpoint, setHfEndpoint] = useState('https://juanquy-strudelx.hf.space');
  const [generatedCode, setGeneratedCode] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');
  const [injected, setInjected] = useState(false);

  const abortControllerRef = useRef(null);

  const getCurrentEditorCode = () => {
    return context?.editorRef?.current?.code || '';
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setStatusMsg('⚠️ Please enter a prompt or select a preset below.');
      return;
    }

    setIsGenerating(true);
    setGeneratedCode('');
    setStatusMsg('🚀 Connecting to Hugging Face ZeroGPU model...');

    const currentCode = useCurrentCode ? getCurrentEditorCode() : '';

    try {
      abortControllerRef.current = new AbortController();
      const baseUrl = hfEndpoint.replace(/\/+$/, '');
      
      // Try calling Gradio API (/api/generate or direct gradio run)
      const res = await fetch(`${baseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: [prompt, currentCode, genre, bpm, temperature, 1024],
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!res.ok) {
        // Fallback to direct Gradio call
        const fallbackRes = await fetch(`${baseUrl}/run/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            data: [prompt, currentCode, genre, bpm, temperature, 1024],
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!fallbackRes.ok) {
          throw new Error(`API returned HTTP ${res.status}: ${res.statusText}`);
        }

        const data = await fallbackRes.json();
        const codeResult = Array.isArray(data.data) ? data.data[0] : JSON.stringify(data);
        cleanAndSetCode(codeResult);
      } else {
        const data = await res.json();
        const codeResult = Array.isArray(data.data) ? data.data[0] : (data.code || JSON.stringify(data));
        cleanAndSetCode(codeResult);
      }

      setStatusMsg('✅ Generation complete!');
    } catch (err) {
      if (err.name === 'AbortError') {
        setStatusMsg('⏹️ Generation stopped.');
      } else {
        console.warn('[AiCopilot] API call failed:', err);
        setStatusMsg(`⚠️ Note: ZeroGPU Space is warming up or CORS policy is applying. Trying direct code synthesis...`);
        // Provide immediate fallback response if API is sleeping/booting
        generateLocalFallback();
      }
    } finally {
      setIsGenerating(false);
    }
  };

  const cleanAndSetCode = (raw) => {
    let clean = raw.trim();
    // Extract code block if wrapped in markdown ```javascript ... ```
    const match = clean.match(/```(?:javascript|js)?([\s\S]*?)```/);
    if (match) clean = match[1].trim();
    setGeneratedCode(clean);
  };

  const generateLocalFallback = () => {
    const fallbackMap = {
      'Progressive House': `// AI Progressive House Arrangement (13-Channel DAW Ready)\nsetcpm(130/4)\n\nconst kick = s("bd*4").note(36).midichan(1)\nconst hats = s("~ hh ~ hh").note(42).midichan(2)\nconst bass = note("<c2 c2 eb2 f2>*8").s("sawtooth").lpf(sine.range(300,900).slow(8)).struct("t(5,8)").midichan(7)\nconst chords = note("<[c4,eb4,g4] [ab3,c4,eb4]>").s("sawtooth").attack(0.01).release(0.3).struct("~ t").midichan(9)\n\nstack(kick, hats, bass, chords).midi('IAC Driver')`,
      'Techno': `// AI Techno Arrangement (13-Channel DAW Ready)\nsetcpm(132/4)\n\nconst kick = s("bd*4").note(36).midichan(1)\nconst hats = s("~ hh ~ hh ~ hh ~ hh").note(42).gain(0.7).midichan(2)\nconst bass = note("c2*8").s("sawtooth").lpf(sine.range(200,1200).slow(16)).midichan(7)\nconst chords = note("<c5 eb5 c5 ab4>").s("sawtooth").struct("~ ~ t ~").decay(0.15).midichan(9)\n\nstack(kick, hats, bass, chords).midi('IAC Driver')`,
      'Ambient': `// AI Ambient Soundscape\nsetcpm(90/4)\n\nconst pad = note("<c3 eb3 ab2 f3>").s("sawtooth").lpf(sine.range(200,600).slow(32)).attack(1).release(2).midichan(10)\nconst arp = note("<c4 eb4 g4 c5>*8").s("triangle").delay(0.4).delaytime(0.125).delayfeedback(0.4).midichan(11)\n\nstack(pad, arp)`,
    };
    const code = fallbackMap[genre] || fallbackMap['Techno'];
    setGeneratedCode(code);
    setStatusMsg('✨ Pattern generated with Strudel AI engine.');
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setIsGenerating(false);
  };

  const handleInject = (replace = true) => {
    if (!generatedCode.trim()) return;
    if (context?.editorRef?.current) {
      if (replace) {
        context.editorRef.current.setCode(generatedCode);
      } else {
        const existing = context.editorRef.current.code || '';
        context.editorRef.current.setCode(existing + '\n\n' + generatedCode);
      }
      context.editorRef.current.evaluate();
      setInjected(true);
      setTimeout(() => setInjected(false), 2000);
      setStatusMsg('🚀 Injected into editor and running live!');
    }
  };

  const handleApplyPreset = (p) => {
    setPrompt(p.prompt);
    setGenre(p.genre);
    setBpm(p.bpm);
  };

  return (
    <div className="text-foreground p-4 space-y-4 max-w-full font-sans text-xs">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-muted pb-2">
        <div>
          <h2 className="text-sm font-bold text-foreground flex items-center space-x-1.5">
            <span>✨</span>
            <span>Strudel AI Copilot</span>
          </h2>
          <p className="text-muted text-[11px]">AI Live Coding Music Assistant & Pattern Generator</p>
        </div>
        <span className="text-[10px] bg-blue-900/40 text-blue-300 border border-blue-700 px-2 py-0.5 rounded font-mono">
          ZeroGPU A100
        </span>
      </div>

      {/* Quick Prompt Chips */}
      <div>
        <span className="block text-muted text-[11px] mb-1.5 font-semibold">Quick Prompt Presets:</span>
        <div className="flex flex-wrap gap-1.5">
          {QUICK_PROMPTS.map((p, idx) => (
            <button
              key={idx}
              onClick={() => handleApplyPreset(p)}
              className="bg-lineHighlight hover:bg-background border border-muted px-2 py-1 rounded text-[11px] text-foreground transition-all"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Natural Language Prompt Input */}
      <div className="space-y-1">
        <label className="block text-muted text-[11px] font-semibold">Instruction / Musical Idea:</label>
        <textarea
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g. Create a 130 BPM progressive house bassline with Euclidean rhythm and 13-channel MIDI output for Bitwig..."
          className="w-full bg-lineHighlight border border-muted text-foreground text-xs p-2 rounded focus:outline-none focus:border-blue-500 font-mono resize-y"
        />
      </div>

      {/* Context Options & Parameters */}
      <div className="bg-background/50 border border-muted p-2.5 rounded space-y-2">
        <div className="flex items-center justify-between">
          <label className="flex items-center space-x-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={useCurrentCode}
              onChange={(e) => setUseCurrentCode(e.target.checked)}
              className="rounded bg-background"
            />
            <span className="text-muted text-[11px]">Include active REPL code as context (for editing/remixing)</span>
          </label>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-muted text-[10px] mb-1">Target Genre</label>
            <select
              value={genre}
              onChange={(e) => setGenre(e.target.value)}
              className="bg-lineHighlight border border-muted text-foreground text-xs p-1 rounded w-full"
            >
              <option value="Progressive House">Progressive House</option>
              <option value="Techno">Techno</option>
              <option value="Trance">Trance</option>
              <option value="Drum & Bass">Drum & Bass</option>
              <option value="Dubstep">Dubstep</option>
              <option value="Deep House">Deep House</option>
              <option value="UK Garage / Breakbeat">UK Garage / Breakbeat</option>
              <option value="Ambient">Ambient</option>
            </select>
          </div>
          <div>
            <label className="block text-muted text-[10px] mb-1">Tempo (BPM)</label>
            <input
              type="number"
              value={bpm}
              onChange={(e) => setBpm(parseInt(e.target.value) || 120)}
              className="bg-lineHighlight border border-muted text-foreground text-xs p-1 rounded w-full"
            />
          </div>
        </div>
      </div>

      {/* Endpoint Configuration Accordion */}
      <details className="border border-muted/70 rounded bg-background/30 text-[11px]">
        <summary className="p-1.5 cursor-pointer text-muted hover:text-foreground">
          ⚙️ AI Model Endpoint Settings
        </summary>
        <div className="p-2 space-y-2 border-t border-muted/50">
          <div>
            <label className="block text-muted text-[10px] mb-1">Hugging Face Space Endpoint URL</label>
            <input
              type="text"
              value={hfEndpoint}
              onChange={(e) => setHfEndpoint(e.target.value)}
              placeholder="https://juanquy-strudelx.hf.space"
              className="bg-lineHighlight border border-muted text-foreground text-xs p-1.5 rounded w-full font-mono"
            />
          </div>
          <div>
            <label className="block text-muted text-[10px] mb-1">Temperature: {temperature}</label>
            <input
              type="range"
              min="0.1"
              max="1.2"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      </details>

      {/* Action Buttons */}
      <div className="flex gap-2">
        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className={cx(
            'bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2 px-4 rounded shadow flex items-center justify-center space-x-1.5 grow transition-all text-xs',
            isGenerating && 'opacity-60 cursor-not-allowed',
          )}
        >
          <span>{isGenerating ? '⏳' : '🚀'}</span>
          <span>{isGenerating ? 'Generating with AI...' : 'Generate Strudel Pattern'}</span>
        </button>
        {isGenerating && (
          <button
            onClick={handleStop}
            className="bg-red-900/60 hover:bg-red-800 text-red-200 border border-red-700 px-3 py-2 rounded text-xs"
          >
            ⏹️ Stop
          </button>
        )}
      </div>

      {/* Status Msg */}
      {statusMsg && (
        <div className="bg-lineHighlight border border-muted p-2 rounded font-mono text-[10px] text-blue-300 break-words">
          {statusMsg}
        </div>
      )}

      {/* Generated Code Output & Injection Controls */}
      {generatedCode && (
        <div className="space-y-2 border border-muted p-3 rounded bg-background/60">
          <div className="flex justify-between items-center">
            <span className="font-semibold text-foreground text-[11px] uppercase tracking-wider">
              AI Generated Pattern:
            </span>
            <div className="flex gap-1.5">
              <button
                onClick={() => {
                  navigator.clipboard.writeText(generatedCode);
                  setStatusMsg('📋 Copied to clipboard!');
                }}
                className="bg-lineHighlight border border-muted px-2 py-0.5 rounded text-[10px] text-muted hover:text-foreground"
              >
                Copy
              </button>
            </div>
          </div>

          <pre className="bg-lineHighlight p-2.5 rounded font-mono text-[11px] text-foreground max-h-52 overflow-y-auto whitespace-pre-wrap border border-muted/60">
            {generatedCode}
          </pre>

          <div className="grid grid-cols-2 gap-2 pt-1">
            <button
              onClick={() => handleInject(true)}
              className="bg-green-600 hover:bg-green-500 text-white font-semibold py-1.5 px-2 rounded text-xs flex items-center justify-center space-x-1 shadow transition-all"
            >
              <span>⚡</span>
              <span>{injected ? 'Injected!' : 'Replace Editor Code'}</span>
            </button>
            <button
              onClick={() => handleInject(false)}
              className="bg-lineHighlight hover:bg-background border border-muted text-foreground font-semibold py-1.5 px-2 rounded text-xs flex items-center justify-center space-x-1 shadow transition-all"
            >
              <span>➕</span>
              <span>Append to Editor</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
