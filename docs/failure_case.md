# AniGraph Failure Case Study

## Case 1: Pronoun Resolution Drift in Multi-Turn Follow-up

### Symptom

User asks "Recommend anime similar to Cowboy Bebop" (returns 5 titles).
Follow-up: "Recommend two more, but not by the same director."

Observed failures:
- Topic drift: "it" resolved to a recommendation result (e.g. Samurai Champloo) instead of the original topic (Cowboy Bebop).
- Count constraint lost: "two" was ignored; system returned 5 titles using default top_k.
- Exclusion constraint wrong: excluded Watanabe works (because Champloo is his), defeating the user intent.

### Root Cause

Three interacting issues:

1. topic_entity polluted by recommendations: recent_entities picks up entities from the recommendation results, and the follow-up round grabs the wrong one.
2. recommendation_count not parsed: The Planner prompt had no explicit instruction to extract numeric constraints from follow-up text.
3. constraints semantics too narrow: constraints dict only supports fixed tags/series exclusions, not dynamic "exclude what was just recommended" semantics.

Relevant source files:
- agents/context_builder.py: topic_entity extraction logic
- agents/state.py: recommendation_count, constraints fields
- agents/planner.py: Planner prompt

### Fix (Implemented)

1. topic_entity maintained independently from recent_entities:
   Set on first identification, only updated on explicit topic switch.
   Recommendation results no longer write to topic_entity.

2. recommendation_count parsed from follow-up text:
   Regex-based extraction at context_builder level.
   Parsed value written to state["recommendation_count"].

3. constraints extended with exclude_previous_recommendations flag.

### Verification

Regression tests added to tests/test_agent.py (MT003, MT008).

### Interview Talking Points

Structure: symptom, root cause (3 layers), fix, remaining gaps.

The most valuable failure I found was pronoun resolution drift in multi-turn follow-up. When a user asked "recommend two more, but not by the same director," the system resolved "it" to a recommendation result instead of the original topic, and both the count and exclusion constraints were lost.

Root cause was three layers: topic_entity polluted by recommendation results, recommendation_count not structurally extracted, and constraints lacking dynamic exclusion semantics.

The fix isolates topic_entity from recommendation results, adds extraction for count constraints, and extends constraints with an exclude-previous flag. I added regression tests for this.

This case also taught me that rule+LLM hybrids break in complex multi-turn scenarios. Long-term, a lightweight LLM should do structured context extraction instead of increasingly complex regex rules.

### Remaining Risks

- topic_entity only supports a single entity (fails on "compare A and B")
- Regex count extraction is fragile ("两三部", "三五部" not covered)
- exclude_previous_recommendations is boolean (cannot express complex filters)
- No golden test set to verify no regression

---

## Case 2: CrossEncoder Silent Degradation

### Symptom

User query "recommend cyberpunk anime similar to Ghost in the Shell" returned low-quality results. Only 1 of top-3 was actually relevant.

### Root Cause

CrossEncoder model file was missing after environment migration. _init_reranker() failed, returned None, and the system silently fell back to raw fusion sort with no warning logged.

Relevant: tools/knowledge_retrieval.py: _reranker_last_failure

### Fix

Added retrieval_errors events to AgentState to record reranker degradation:

```python
state["retrieval_errors"].append({
    "event": "reranker_degraded",
    "reason": "model_unavailable",
    "fallback": "fusion_raw"
})
```

SSE stream now surfaces this event for debugging and monitoring.

### Interview Talking Points

We also hit a silent degradation bug: the CrossEncoder model file went missing after an environment change, and the system fell back to raw fusion sort without a peep. The fix added retrieval_errors events to the state so every degradation decision becomes an observable signal. The lesson: every try/except fallback needs a corresponding observability channel.

---

## Template: Document Your Own Failures

```text
Title: [one-liner]
Symptom: [what the user saw]
Repro steps: [minimal steps]
Root cause: [which module, what design flaw]
Impact scope: [how many users/query types affected]
Fix: [what changed]
Verification: [how you confirmed it is fixed]
Remaining gaps: [what the fix does not cover]
```
