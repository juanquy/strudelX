import os
import gradio as gr
from fastapi.staticfiles import StaticFiles

# Minimal lightweight Gradio app satisfying Hugging Face SDK
with gr.Blocks(title="Strudel REPL") as demo:
    gr.HTML("""
    <iframe
      src="/strudel_app/index.html"
      style="position:fixed; top:0; left:0; width:100vw; height:100vh; border:none; z-index:999999; background:#0b0b0d;"
      allow="autoplay *; sound-active *; audio-capture *; midi *; microphone *; speaker-selection *; clipboard-read *; clipboard-write *"
    ></iframe>
    """)

app = gr.routes.App.create_app(demo)

DIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "website", "dist"))

if os.path.exists(DIST_DIR):
    app.mount("/strudel_app", StaticFiles(directory=DIST_DIR, html=True), name="strudel_app")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
