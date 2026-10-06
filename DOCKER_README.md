# Docker deployment

This packaging requires the standalone Python application files, requirements.txt and tests/ in the repository root. Packaging alone cannot build until those files are committed. The original notebook and reports remain unchanged.

## Setup
Use Docker Engine or Docker Desktop with Linux containers and Compose v2. Create a local .env containing OPENAI_API_KEY and your model configuration. Never commit credentials. Keep PDFs in documents/.

```bash
docker compose build
docker compose run --rm ingest
docker compose up -d chatbot
docker compose logs -f chatbot
```

Open http://localhost:7860. Ingestion performs paid embedding API calls and is deliberately not automatic. Cache storage is a Docker named volume, separate from host data/.

## Validation
```bash
docker compose ps
docker compose run --rm --no-deps chatbot python -m pytest -q
docker compose run --rm evaluate
```

Evaluation output is stored in the evaluation_results volume. The HTTP health check verifies UI availability, not API access or answer correctness. Image build and live integration have not been tested by the assistant.

## Updates
Stop the chatbot before changing PDFs or rebuilding the cache:

```bash
docker compose stop chatbot
docker compose run --rm ingest
docker compose up -d chatbot
```

Code or dependency changes require rebuilding the image. Stop with docker compose down. Named volumes persist. docker compose down -v deletes cached embeddings and evaluation results.

## Security and limitations
Runs as non-root UID 10001, drops capabilities, uses a read-only root filesystem and publishes only to localhost. PDFs and chatbot cache mounts are read-only. Gradio binds 0.0.0.0 inside the image through docker_bind.py; host source remains unchanged. Do not expose publicly without authentication and TLS. Container administrators can inspect runtime environment variables. PDF text and queries are still sent to OpenAI. Dependency ranges are not a tested lockfile; pin resolved dependencies and image digest after validation. Research prototype, not operational aviation guidance.
