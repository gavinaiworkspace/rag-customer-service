"""RAG chain: retrieve top-K chunks from Chroma, generate a grounded answer.

Library use (for the future Streamlit UI / eval harness):

    from chain import build_chain, get_retriever
    chain = build_chain()                        # defaults to v1
    chain = build_chain(prompt_version="v2")     # use a specific version
    answer = chain.invoke("What vehicles do you have?")

CLI smoke test:

    python chain.py "What vehicles do you have?"
    python chain.py "What vehicles do you have?" --prompt v2
    python chain.py                # uses the default question with v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import tiktoken
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config

PromptVersion = Literal["v1", "v2", "v3", "v4", "v5"]
SUPPORTED_VERSIONS: tuple[str, ...] = ("v1", "v2", "v3", "v4", "v5")


def _load_prompt(version: str) -> ChatPromptTemplate:
    """Import and return the PROMPT object for the requested version."""
    if version == "v1":
        from prompts.v1_baseline import PROMPT
    elif version == "v2":
        from prompts.v2_cot import PROMPT
    elif version == "v3":
        from prompts.v3_fewshot import PROMPT
    elif version == "v4":
        from prompts.v4_guardrails import PROMPT
    elif version == "v5":
        from prompts.v5_optimised import PROMPT
    else:
        raise ValueError(
            f"Unknown prompt version '{version}'. "
            f"Choose from: {', '.join(SUPPORTED_VERSIONS)}"
        )
    return PROMPT

DEFAULT_QUESTION = "What services do you offer?"
DEFAULT_PROMPT_VERSION = "v1"


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


def build_chain(prompt_version: str = DEFAULT_PROMPT_VERSION) -> Runnable:
    """Compose the LCEL chain: retrieve -> format context -> prompt -> LLM -> string."""
    config.require_api_key()

    retriever = get_retriever()
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
    )
    prompt = _load_prompt(prompt_version)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_chain_with_sources(prompt_version: str = DEFAULT_PROMPT_VERSION) -> Runnable:
    """Same as build_chain() but also exposes retrieved docs in the output.

    Returns a runnable whose output dict has keys: 'answer', 'context_docs', 'context_text'.
    Used by the CLI smoke test so we can show which sources were retrieved.
    """
    config.require_api_key()

    retriever = get_retriever()
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
    )
    prompt = _load_prompt(prompt_version)

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
        | prompt
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


def _cli(question: str, prompt_version: str = DEFAULT_PROMPT_VERSION) -> int:
    chain = build_chain_with_sources(prompt_version)
    print(f"Prompt version : {prompt_version}")
    print(f"Question       : {question}\n")

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
    parser = argparse.ArgumentParser(description="Query the Chauffeur For All RAG pipeline.")
    parser.add_argument("question", nargs="*", help="Question to ask (default: sample question)")
    parser.add_argument(
        "--prompt",
        choices=SUPPORTED_VERSIONS,
        default=DEFAULT_PROMPT_VERSION,
        help="Prompt version to use (default: v1)",
    )
    args = parser.parse_args()
    question = " ".join(args.question).strip() or DEFAULT_QUESTION
    sys.exit(_cli(question, args.prompt))
