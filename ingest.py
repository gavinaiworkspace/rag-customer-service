"""Document ingestion: load docs/, split into token-bounded chunks, embed, persist to Chroma.

Run:
    python ingest.py

Behaviour:
- Walks `docs/` for .txt and .pdf files (extensions configured in config.py)
- Splits with a tiktoken-backed RecursiveCharacterTextSplitter so chunk_size and
  chunk_overlap are measured in real tokens, not characters
- Embeds with OpenAI's text-embedding-3-small
- Persists to a local Chroma collection configured for cosine similarity
- Prints chunk and token statistics so the OpenAI budget can be monitored
"""

from __future__ import annotations

import sys
from pathlib import Path

import tiktoken
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def discover_documents(docs_dir: Path) -> list[Path]:
    """Return every supported file under docs_dir, sorted for stable ingestion order."""
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"docs directory not found: {docs_dir}. "
            "Create it and add .txt or .pdf business documents before running ingest."
        )
    files: list[Path] = []
    for ext in config.SUPPORTED_EXTENSIONS:
        files.extend(docs_dir.rglob(f"*{ext}"))
    return sorted(files)


def load_document(path: Path) -> list[Document]:
    """Dispatch to the right loader based on file extension."""
    ext = path.suffix.lower()
    if ext == ".txt":
        # autodetect encoding so we don't choke on UTF-8 BOM / cp1252 files on Windows
        return TextLoader(str(path), autodetect_encoding=True).load()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    raise ValueError(f"Unsupported file extension: {ext}")


def count_tokens(text: str, model: str) -> int:
    """Count tokens with the encoder matching the embedding model when possible."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def main() -> int:
    config.require_api_key()

    docs_dir = Path(config.DOCS_DIR)
    files = discover_documents(docs_dir)
    if not files:
        print(
            f"No supported documents found under {docs_dir}. "
            f"Add .txt or .pdf files and re-run."
        )
        return 1

    print(f"Found {len(files)} document(s) under {docs_dir}:")
    for f in files:
        print(f"  - {f.relative_to(config.PROJECT_ROOT)}")

    raw_docs: list[Document] = []
    for f in files:
        loaded = load_document(f)
        # Stamp a stable, human-readable source so retrieval results are traceable
        # back to the originating file rather than an absolute path.
        for d in loaded:
            d.metadata["source"] = f.name
        raw_docs.extend(loaded)
    print(f"Loaded {len(raw_docs)} raw document section(s).")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)
    if not chunks:
        print("Splitter produced 0 chunks; check your documents.")
        return 1

    total_tokens = sum(count_tokens(c.page_content, config.EMBEDDING_MODEL) for c in chunks)
    avg_tokens = total_tokens / len(chunks)
    print(
        f"Split into {len(chunks)} chunk(s) "
        f"(target {config.CHUNK_SIZE} tokens, overlap {config.CHUNK_OVERLAP}; "
        f"avg {avg_tokens:.0f} tokens, total {total_tokens} tokens to embed)."
    )

    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )

    print(f"Embedding and persisting to {config.CHROMA_DIR} ...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
        # Chroma defaults to L2; the brief requires cosine similarity.
        collection_metadata={"hnsw:space": config.DISTANCE_METRIC},
    )

    print(
        f"Done. Collection '{config.COLLECTION_NAME}' "
        f"now contains {len(chunks)} chunk(s) at {config.CHROMA_DIR}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
