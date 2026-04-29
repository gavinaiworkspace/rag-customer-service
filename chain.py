"""RAG chain: retrieve top-K chunks from Chroma, generate a grounded answer with the v1 prompt.

Library use (for the future Streamlit UI / eval harness):

    from chain import build_chain, get_retriever
    chain = build_chain()
    answer = chain.invoke("What vehicles do you have?")

CLI smoke test:

    python chain.py "What vehicles do you have?"
    python chain.py                # uses the default sample question
"""

from __future__ import annotations

import sys
from pathlib import Path

import tiktoken
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config
from prompts.v1_baseline import PROMPT

DEFAULT_QUESTION = "What services do you offer?"


def _ensure_chroma_dir() -> None:
    """Fail fast with an actionable message if the user hasn't run ingest yet."""
    if not Path(config.CHROMA_DIR).exists():
        raise FileNotFoundError(
            f"Chroma store not found at {config.CHROMA_DIR}. "
            "Run `python ingest.py` first to build the vector store."
        )


def get_vectorstore() -> Chroma:
    """Open the persisted Chroma collection that ingest.py created."""
    _ensure_chroma_dir()
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    )


def get_retriever():
    """Cosine-similarity retriever returning the top-K chunks per query."""
    return get_vectorstore().as_retriever(search_kwargs={"k": config.TOP_K})


def format_docs(docs: list[Document]) -> str:
    """Stitch retrieved chunks into a single context block, tagged with their source file."""
    if not docs:
        return "(no relevant context retrieved)"
    blocks: list[str] = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        blocks.append(f"[Source {i}: {source}]\n{d.page_content.strip()}")
    return "\n\n".join(blocks)


def build_chain() -> Runnable:
    """Compose the LCEL chain: retrieve -> format context -> v1 prompt -> LLM -> string."""
    config.require_api_key()

    retriever = get_retriever()
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
    )

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )


def build_chain_with_sources() -> Runnable:
    """Same chain as build_chain() but also exposes the retrieved docs in the output.

    Returns a runnable whose output dict has keys: 'answer', 'context_docs', 'context_text'.
    Used by the CLI smoke test so we can show the user which sources were retrieved.
    """
    config.require_api_key()

    retriever = get_retriever()
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
    )

    retrieve_step = RunnableParallel(
        question=RunnablePassthrough(),
        context_docs=retriever,
    )

    def _attach_context_text(state: dict) -> dict:
        return {**state, "context_text": format_docs(state["context_docs"])}

    answer_step = (
        {
            "context": lambda s: s["context_text"],
            "question": lambda s: s["question"],
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return retrieve_step | _attach_context_text | RunnableParallel(
        answer=answer_step,
        context_docs=lambda s: s["context_docs"],
        context_text=lambda s: s["context_text"],
    )


def _count_tokens(text: str, model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _cli(question: str) -> int:
    chain = build_chain_with_sources()
    print(f"Question: {question}\n")

    result = chain.invoke(question)
    docs: list[Document] = result["context_docs"]
    context_text: str = result["context_text"]
    answer: str = result["answer"]

    print(f"Retrieved {len(docs)} chunk(s):")
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        print(f"  [{i}] {source}: {snippet}")
    print()

    print("Answer:")
    print(answer)
    print()

    prompt_tokens = _count_tokens(context_text + question, config.LLM_MODEL)
    completion_tokens = _count_tokens(answer, config.LLM_MODEL)
    print(
        f"Token usage (approx): prompt={prompt_tokens}, completion={completion_tokens}, "
        f"total={prompt_tokens + completion_tokens}"
    )
    return 0


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    sys.exit(_cli(q))
