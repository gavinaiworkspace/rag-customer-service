# Luxury Transport RAG Customer Service Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers customer enquiries about a
luxury transport business by retrieving information from approved business documents
(services, vehicles, pricing, policies, FAQs) and grounding LLM responses in that context.

University CPD project (COMP4060) — 35 hours over 7 weeks.

## Tech stack

- **Python 3.11+**
- **LangChain 0.3** for RAG pipeline orchestration (LCEL)
- **ChromaDB** for the local persisted vector store (cosine similarity)
- **OpenAI** — `text-embedding-3-small` for embeddings, `gpt-4o-mini` for generation
  (swap to `gpt-4o` for the final evaluation run)
- **tiktoken** for token-budget monitoring
- **Streamlit** for the chat UI _(later milestone)_
- **RAGAS** for evaluation metrics _(later milestone)_

## Project structure

```
luxury-transport-rag/
├── app.py                 # Streamlit UI (later milestone)
├── chain.py               # RAG chain (retrieval + generation) + CLI smoke test
├── config.py              # API key loading, model + chunking + retrieval params
├── ingest.py              # Load docs/, split, embed, persist to Chroma
├── prompts/
│   ├── __init__.py
│   ├── v1_baseline.py     # Initial system prompt
│   ├── v2_cot.py          # Chain-of-thought (later milestone)
│   ├── v3_fewshot.py      # Few-shot examples (later milestone)
│   ├── v4_guardrails.py   # Off-topic guardrails (later milestone)
│   └── v5_optimised.py    # Final optimised version (later milestone)
├── eval/                  # RAGAS evaluation harness (later milestone)
├── docs/                  # Business documents (.txt and .pdf)
├── prompt_log.md          # Design rationale per prompt version (later milestone)
├── requirements.txt
├── .env.example
└── README.md
```

## Quickstart

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure your API key

Copy `.env.example` to `.env` and paste your OpenAI key:

```
OPENAI_API_KEY=sk-...
```

### 3. Add business documents

Drop `.txt` or `.pdf` files describing your services, fleet, pricing, and policies
into the `docs/` folder. A tiny `docs/sample.txt` stub ships with the project so the
pipeline runs out of the box — replace it with your real documents.

### 4. Build the vector store

```bash
python ingest.py
```

This loads every supported file under `docs/`, splits it into ~800-token chunks with
100-token overlap, embeds the chunks, and persists them to `chroma_db/` using cosine
similarity. It also prints chunk and token counts so you can keep an eye on spend.

### 5. Ask a question

```bash
python chain.py "What vehicles do you have?"
```

The CLI prints the retrieved source chunks and a grounded answer using the v1 baseline
prompt, plus a token-usage report.

## Configuration

All tunable parameters live in `config.py`:

| Setting           | Default                  | Notes                                        |
| ----------------- | ------------------------ | -------------------------------------------- |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embeddings                            |
| `LLM_MODEL`       | `gpt-4o-mini`            | Swap to `gpt-4o` for final evaluation        |
| `CHUNK_SIZE`      | `800` tokens             | Within the 500-1000 spec range               |
| `CHUNK_OVERLAP`   | `100` tokens             |                                              |
| `TOP_K`           | `3`                      | Top-K chunks retrieved per query             |
| `CHROMA_DIR`      | `chroma_db`              | Persisted vector store directory             |
| `DOCS_DIR`        | `docs`                   | Source documents directory                   |

## Roadmap

- [x] **Milestone 1** — Project scaffold, ingest pipeline, baseline RAG chain, CLI smoke test
- [ ] Milestone 2 — Prompt versions v2-v5 + `prompt_log.md` rationale
- [ ] Milestone 3 — RAGAS evaluation harness with comparison chart
- [ ] Milestone 4 — Streamlit chat UI with conversation history sidebar
