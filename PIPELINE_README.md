# Standalone aviation Self-RAG pipeline

Python implementation alongside the original notebook. Source modules live at the repository root and reuse documents/. The notebook and original README are preserved. Docker packaging remains on feature/docker-chatbot and is not part of this commit.

## Setup
Use Python 3.11 or 3.12.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy .env.example to .env and set OPENAI_API_KEY locally. Choose a chat model available to your account that supports structured outputs. Never commit credentials. Reports must be text-readable PDFs; OCR is not implemented.

```bash
python ingest.py
python chatbot.py
python cli.py "What was the probable cause of the VT-PTE accident?" --trace
python -m pytest -q
python evaluate.py
```

Open the local address printed by Gradio. Ingestion performs paid embedding API calls. Re-run ingestion when PDFs, embedding model or chunk settings change. Cache files are JSON and non-pickle NumPy arrays.

## Workflow
Question and recent conversation -> standalone rewrite -> routing -> FAISS retrieval -> relevance grading -> cited structured generation -> grounding check -> one retry by default -> answer or abstention. Greetings and unrelated questions receive fixed responses. No web fallback is included.

Aircraft identifiers and preliminary/final status are inferred from filenames and must be verified against reports. Citations identify excerpt IDs and physical PDF pages, not printed page labels. Conversation history is session-local and only the last six messages are used for rewriting.

## Validation and limitations
The generated source was syntax-checked, but API execution, dependency installation and UI integration have not been validated by the assistant. Tests cover citation rendering, not semantic correctness. Evaluation writes review records to runs/evaluation.json; manually check answers against reports. Model self-checks do not prove accuracy or eliminate hallucinations.

This is Self-RAG-inspired prompted orchestration, not the original paper's trained reflection-token implementation. No authentication, OCR, robust table extraction or production hardening is included. OpenAI receives extracted text, retrieved excerpts, queries and recent conversation. Do not use as operational aviation guidance.
