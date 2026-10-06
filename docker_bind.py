from pathlib import Path

path = Path("chatbot.py")
text = path.read_text(encoding="utf-8")
old = 'launch(server_name="127.0.0.1", share=False)'
new = 'launch(server_name="0.0.0.0", server_port=7860, share=False)'
if old not in text:
    raise RuntimeError("chatbot.py launch signature changed; update docker_bind.py.")
path.write_text(text.replace(old, new), encoding="utf-8")
