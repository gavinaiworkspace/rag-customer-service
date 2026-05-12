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

### What changed from v2

v3 removes the explicit reasoning block and instead adds three example
question-answer pairs showing the desired answer style:

1. A factual fleet answer.
2. A pricing answer that keeps the "indicative only" caveat.
3. A missing-information answer that declines to invent package inclusions.

### Why this change was made

The first RAGAS run showed answer relevancy was moderate and v2 did not improve
over v1. Few-shot examples give the model concrete patterns for direct,
customer-ready answers, which should reduce vague or padded responses.

### Expected improvement

| Metric | Expected change | Reasoning |
|---|---|---|
| Faithfulness | Slight increase | The decline example reinforces not guessing when details are absent |
| Answer relevance | Increase | Examples demonstrate the desired level of detail and directness |
| Context precision | Neutral | Retrieval is unchanged |

---

## v4 — Guardrails (`prompts/v4_guardrails.py`)

### What changed from v3

v4 adds explicit guardrails:

1. Never infer information not directly stated in the context.
2. Do not use outside knowledge.
3. Decline or offer escalation when the documents do not cover the question.
4. Answer partial questions only where the context supports them.
5. Do not confirm specific locations, inclusions, prices, or availability unless
   directly stated.
6. Decline off-topic requests politely.

### Why this change was made

Faithfulness was the largest gap in the first evaluation run. The model needs
stronger instructions not to over-confirm edge cases such as service areas or
unstated package inclusions.

### Expected improvement

| Metric | Expected change | Reasoning |
|---|---|---|
| Faithfulness | Increase | Hard grounding rules reduce unsupported claims |
| Answer relevance | Slight increase | Partial-answer rules keep responses aligned with the exact question |
| Context precision | Neutral | Retrieval is unchanged |

---

## v5 — Optimised (`prompts/v5_optimised.py`)

### What changed from v4

v5 combines the strongest elements from the earlier prompts:

1. A short silent context-coverage checklist inspired by v2.
2. Few-shot answer patterns from v3.
3. Strict guardrails from v4.

It avoids the verbose visible `<thinking>` format because the first evaluation
did not show an improvement from v2 over v1.

### Why this change was made

The evaluation suggested the main target should be faithfulness first, with
answer relevance second. v5 therefore prioritises strict grounding, direct
answers, and clear declines when the retrieved documents do not contain enough
information.

### Expected improvement

| Metric | Expected change | Reasoning |
|---|---|---|
| Faithfulness | Increase | Combines context checking with explicit no-inference rules |
| Answer relevance | Increase | Examples and direct-answer structure keep the answer focused |
| Context precision | Neutral | Retrieval is unchanged |
