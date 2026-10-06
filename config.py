import os
from pathlib import Path
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
DOCUMENTS = BASE / "documents"
INDEX = BASE / "data" / "index"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError("Require 0 <= CHUNK_OVERLAP < CHUNK_SIZE")
if RETRIEVAL_K < 1 or MAX_RETRIES < 0:
    raise ValueError("Invalid retrieval or retry configuration")
