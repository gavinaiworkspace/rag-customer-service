"""Streamlit chat UI for the Chauffeur For All RAG assistant.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from chain import build_chain

PROMPT_DESCRIPTIONS = {
    "v1": "Baseline grounded prompt with a concise customer-service style.",
    "v2": "Adds an internal context-checking step before answering.",
    "v3": "Uses few-shot examples to model strong, relevant answers.",
    "v4": "Adds strict guardrails to reduce unsupported claims.",
    "v5": "Optimised version combining context checks, examples, and guardrails.",
}
PROMPT_VERSIONS = tuple(PROMPT_DESCRIPTIONS)
WELCOME_MESSAGE = "Welcome to Chauffeur For All. How can I assist you today?"
ERROR_MESSAGE = (
    "Sorry, I couldn't generate a response just now. "
    "Please try again, or contact the Chauffeur For All team for help."
)


st.set_page_config(
    page_title="Chauffeur For All — AI Assistant",
    page_icon="🚘",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_chain(prompt_version: str):
    """Build and cache one RAG chain per prompt version."""
    return build_chain(prompt_version=prompt_version)


def initialise_chat_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ]


def render_sidebar() -> str:
    st.sidebar.title("Chauffeur For All")
    st.sidebar.caption("Luxury chauffeur services across Sydney")

    prompt_version = st.sidebar.selectbox(
        "Prompt version",
        PROMPT_VERSIONS,
        index=PROMPT_VERSIONS.index("v5"),
        help="Changing this keeps the chat history and uses the new prompt for the next message.",
    )

    st.sidebar.markdown("### Prompt Versions")
    for version, description in PROMPT_DESCRIPTIONS.items():
        st.sidebar.markdown(f"**{version}**: {description}")

    return prompt_version


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def main() -> None:
    initialise_chat_history()
    prompt_version = render_sidebar()

    st.title("Chauffeur For All — AI Assistant")
    st.caption("Ask about services, vehicles, pricing, policies, and service areas.")

    render_chat_history()

    user_question = st.chat_input("Ask a question about Chauffeur For All")
    if not user_question:
        return

    append_message("user", user_question)
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                chain = get_chain(prompt_version)
                answer = chain.invoke(user_question)
            except Exception:
                answer = ERROR_MESSAGE

        st.markdown(answer)

    append_message("assistant", answer)


if __name__ == "__main__":
    main()
