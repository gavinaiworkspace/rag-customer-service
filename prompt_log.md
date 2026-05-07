# Prompt Engineering Log

Design rationale for each prompt version in the Chauffeur For All RAG pipeline.
Each entry records what changed, why it was changed, and the expected effect on
RAGAS evaluation metrics (faithfulness, answer relevance, context precision).

---

## v1 — Baseline (`prompts/v1_baseline.py`)

### What it does

Minimal system prompt with three elements:
- Role statement: customer service assistant for a luxury transport business
- Grounding rule: answer using only the retrieved context; say you don't know if context is insufficient
- Style guide: professional but warm, concise

### Why this is the starting point

The baseline is intentionally bare-bones. Its job is to establish a performance
floor against which subsequent versions can show measurable improvement. Adding
anything beyond the minimum would compress the headroom for v2–v5.

### Expected RAGAS baseline

| Metric | Expectation |
|---|---|
| Faithfulness | Moderate — the model may still occasionally elaborate beyond context |
| Answer relevance | Moderate — no mechanism to focus the model on the specific question |
| Context precision | Not directly affected by the prompt |

---

## v2 — Chain-of-Thought (`prompts/v2_cot.py`)

### What changed from v1

A `<thinking>` reasoning block was added to the system prompt, placed before the
customer-facing answer. The model is instructed to:

1. Identify the specific sentences or bullet points in the retrieved context that
   directly address the customer's question.
2. Note any gaps — parts of the question not covered by the context.
3. Decide what to include in the answer based only on what step 1 found relevant.

The `<thinking>` block uses XML-style tags so it can be stripped or logged
separately. All other elements (tone, grounding rule, style) are identical to v1,
making CoT the only changed variable.

### Why this change was made

v1 asks the model to jump straight from raw context to a customer answer. This
means the model can silently pick the wrong part of the context, blend information
from multiple chunks incorrectly, or fill gaps with plausible-sounding fabrications.

By forcing an explicit reasoning step, the model must surface which context
sentences it is relying on before it writes the answer. This creates an internal
accountability mechanism: if the relevant sentences aren't in the context, the gap
becomes visible in step 2, which makes the model more likely to acknowledge the
limitation rather than guess.

The business name ("Chauffeur For All") was also added to the role statement in v2
to ground the model's identity more precisely.

### Expected improvement over v1

| Metric | Expected change | Reasoning |
|---|---|---|
| Faithfulness | Increase | The model must explicitly check context coverage before answering, reducing unsupported claims |
| Answer relevance | Increase | The reasoning step focuses the model on the exact question asked, reducing off-topic padding |
| Context precision | Neutral | Retrieval is unchanged; precision depends on the retriever, not the generation prompt |

---

## v3 — Few-Shot Examples (`prompts/v3_fewshot.py`)

_To be completed after v3 is implemented and evaluated._

---

## v4 — Guardrails (`prompts/v4_guardrails.py`)

_To be completed after v4 is implemented and evaluated._

---

## v5 — Optimised (`prompts/v5_optimised.py`)

_To be completed after v5 is implemented and evaluation results are available._
