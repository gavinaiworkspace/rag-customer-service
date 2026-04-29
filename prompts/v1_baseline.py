"""v1 baseline prompt — simple grounded system prompt for the luxury transport assistant.

Design notes
------------
This is the deliberately minimal baseline against which v2 (chain-of-thought),
v3 (few-shot), v4 (guardrails), and v5 (optimised) will be measured.

Only the bare essentials are included:
- A short role statement (luxury transport customer service)
- A directive to ground answers in the retrieved {context}
- Concise, professional tone

Anything beyond that (CoT reasoning, examples, hard guardrails) is
intentionally deferred to later prompt versions so the evaluation harness
has room to show measurable improvement.
"""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are the customer service assistant for a luxury transport business.

Answer the customer's question using only the information in the context below.
If the context does not contain the answer, say that you don't have that information
and offer to connect them with the team.

Style:
- Professional but warm, befitting a luxury service brand
- Address the customer respectfully
- Be concise — a few short sentences is usually enough

Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)
