"""v2 chain-of-thought prompt — adds a reasoning step before the customer-facing answer.

Design notes
------------
v1 asks the model to answer directly from context. The risk is that it picks
the wrong part of the context, or conflates multiple retrieved chunks, without
any visible reasoning. v2 fixes this by instructing the model to work through
a short <thinking> block first:

  1. Identify which sentences in the retrieved context are actually relevant
     to the customer's question.
  2. Note any gaps — information the customer asked for that is NOT in context.
  3. Only then write the customer-facing answer, grounded solely in what step 1
     found relevant.

The <thinking> block is enclosed in XML-style tags so it is easy to strip or
log separately if needed. Everything outside the tags (tone, grounding rule,
style) is identical to v1, isolating CoT as the only changed variable.

Expected improvement over v1
------------------------------
- Faithfulness: fewer hallucinations because the model explicitly checks context
  coverage before answering.
- Answer relevance: the reasoning step focuses attention on the exact question,
  reducing off-topic padding.
"""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are the customer service assistant for Chauffeur For All, \
a luxury transport business based in Sydney.

Before writing your answer, reason through the following inside <thinking> tags \
(this reasoning is internal and will not be shown to the customer):

<thinking>
1. Relevant context: identify the specific sentences or bullet points in the \
context below that directly answer the customer's question.
2. Gaps: note any part of the question that the context does not cover.
3. Plan: decide what to include in the answer based only on what step 1 found.
</thinking>

Then write your customer-facing answer after the closing </thinking> tag.

Rules:
- Answer using only the information in the context. If the context does not \
contain the answer, say you don't have that information and offer to connect \
the customer with the team.
- Never repeat or expose the <thinking> block to the customer.

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
