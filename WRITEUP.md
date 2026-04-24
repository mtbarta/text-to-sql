## Summary

The agent passes 53% of hard cases given to us. We've found additional areas of investment to make in retrieval based on relevance metrics. We've also started to identify problematic agent behaviors that lead to poor results.

The work broke down into four iterations:

1. Baseline on easy cases to validate the agent and build evaluation tooling
2. Baseline on hard cases to measure the gap and understand failure modes
3. Upgraded model to improve iteration behavior
4. Measured retrieval quality to confirm where the remaining gap lives

The clearest finding: retrieval is the bottleneck. The agent is not finding the right markdown rules, and that's driving most wrong-filter failures. The next meaningful improvement will come from fixing the retrieval pipeline.

---

## Building evaluation tooling first

Before spending time tuning the agent, I invested in an analysis script that aggregates failures.

The script produces two types of signal:

**Deterministic behavioral insights** examine the agent's tool call trajectory. We know a well-functioning agent should run `run_query` to validate its SQL before submitting, and should iterate when the first result looks wrong. Cases where the agent runs a single query and immediately submits are a red flag. This is cheap to compute and easy to act on.

**LLM-judged failure categories** classify each mismatch into a specific reason: wrong filter, wrong groupby, wrong column, wrong aggregation, etc. These are less precise than deterministic checks, but they're far more informative than a bare `MISMATCH` label. 

---

## Iteration 1: Easy Baseline

**73.4% pass rate** (47/64 passed)
[experiment analysis](experiment_analysis/run_20260423_021111)

The easy dataset tests SQL generation without any markdown rules. The goal here was to understand how well the agent could navigate the database schema on its own.

I gave the agent granular tools to list and describe schemas and tables rather than injecting that context directly into the prompt. I wanted to see if the model would use them correctly, and let failures guide what to fix.

| Failure Type | Count | % of failures |
|---|---|---|
| MISMATCH | 12 | 71% |
| NO_SUBMISSION | 3 | 18% |
| AGENT_ERROR | 2 | 12% |

53% of failures came from single-query trajectories. The agent ran one query and submitted without any validation or iteration. 

The LLM judge broke down the 12 mismatches:

| Category | Count |
|---|---|
| wrong_filter | 4 |
| wrong_groupby | 3 |
| wrong_column | 2 |
| null_handling | 2 |
| wrong_schema | 1 |

These failures gave us concrete prompt improvements. For example, the agent was adding defensive `IS NULL OR col <> value` guards and spurious filters like `Cancelled=0` that the question never asked for. We added explicit instructions to suppress these patterns:

```
- Do NOT add 'IS NULL OR col <> value' guards unless the question explicitly
  requires special NULL handling.
- Do NOT add filters that aren't mentioned in the question.
- When grouping by a name column, always also include the corresponding ID
  column in GROUP BY to avoid incorrect aggregation.
- Return exactly the columns the question requests — no extras.
```

Ideally, I'd spend more time here and achieve >85% pass rate. However, we have limited time and were able to make meaninful improvements.

---

## Iteration 2: Hard Baseline

**42.2% pass rate** (27/64 passed)
[experiment analysis](experiment_analysis/run_20260423_030920)

The hard dataset introduces markdown files containing domain-specific rules the agent must follow. Our first job is to determine how to retrieve these rules.

### Retrieval design

I considered three approaches:

**Option A — Stuff everything into context.** Simple, but noisy. My experience is that models degrade when context is cluttered with irrelevant rules. It's also harder to debug when things go wrong.

**Option B — Lexical search (BM25).** Simple to build and easy to inspect. The risk is vocabulary mismatch. If the agent doesn't use the right keywords, it won't find the right rules. However, higher precision techniques are a good place to start from and build on.

**Option C — Semantic search.** Avoids vocabulary mismatch but always returns results, which makes precision harder to control. More work to debug and improve.

I chose **Option B**. A lexical index is transparent, easy to evaluate, and lets us improve systematically. Each chunk is formed from the document name, section header, and rule text combined, and so a query matching any of those terms will retrieve the rule.

### Results

**Behavioral Insights (failures only)**

| Insight Code | Count | % of failures |
|---|---|---|
| single_query_no_iteration | 15 | 41% |
| no_query_testing | 5 | 14% |

**LLM Judge Category Distribution (35 MISMATCH cases)**

| Category | Count | % of judged |
|---|---|---|
| wrong_filter | 17 | 49% |
| wrong_groupby | 5 | 14% |
| wrong_aggregation | 5 | 14% |
| wrong_column | 5 | 14% |
| other | 1 | 3% |
| null_handling | 1 | 3% |
| over_engineering | 1 | 3% |

The behavioral analysis shows 41% of failures were single-query trajectories. The LLM judge showed wrong_filter had jumped to 49% of mismatches, up from 33% on easy. These are almost certainly rules the agent needed but didn't retrieve.

This pointed to two problems to tackle: the agent wasn't iterating enough, and retrieval wasn't finding the right rules.

---

## Iteration 3: Better Model

**50.8% pass rate** (32/63 passed)
[experiment analysis](experiment_analysis/run_20260423_053812)

Before rewriting the prompt, I wanted to test whether a stronger model would naturally iterate more. If the iteration problem was the model's capability, swapping the model would show that. If it was a prompt problem, we'd still see single-query trajectories after the swap.

I switched to [Moonshot kimi-k2.6](https://llm-benchmark.tinybird.live/), which benchmarks well on SQL generation and is cost-competitive with other capable models. I also added a single prompt instruction:

```
- Validate that your SQL query respects all the rules and requirements before submitting.
```

### Results

**Behavioral Insights (failures only)**

| Insight Code | Count | % of failures |
|---|---|---|
| single_query_no_iteration | 2 | 6% |

**LLM Judge Category Distribution (25 MISMATCH cases)**

| Category | Count | % of judged |
|---|---|---|
| wrong_filter | 14 | 56% |
| wrong_column | 5 | 20% |
| wrong_aggregation | 3 | 12% |
| wrong_groupby | 3 | 12% |

**Single-query trajectories dropped from 41% to 6% of failures.** The model now iteratively validates the query. But the pass rate only improved from 42% to 51%, and wrong_filter still accounts for 56% of mismatches. The agent is trying harder, but it's still not finding the right rules to apply.

This confirmed the retrieval hypothesis: iteration behavior was a model capability issue, but the filter failures have a different root cause.

---

## Iteration 4: Measuring Retrieval Quality

**53.1% pass rate** (34/64 passed)
[experiment analysis](experiment_analysis/run_20260424_043219)

With the iteration problem resolved, I wanted to quantify retrieval quality directly. I used an LLM judge to classify each retrieved snippet as relevant or not relevant to the question, then computed Precision@5.

I also made a few small prompt tweaks to encourage the agent to search more proactively.

### Results

**Behavioral Insights (failures only)**

| Insight Code | Count | % of failures |
|---|---|---|
| single_query_no_iteration | 4 | 13% |
| repeated_query_errors | 1 | 3% |

**LLM Judge Category Distribution (23 MISMATCH cases)**

| Category | Count | % of judged |
|---|---|---|
| wrong_filter | 11 | 48% |
| wrong_aggregation | 5 | 22% |
| wrong_column | 4 | 17% |
| wrong_groupby | 2 | 9% |
| other | 1 | 4% |

| Metric | Value |
|---|---|
| Zero-result rate | 0.0% |
| Precision@5 (LLM-judged) | 16.1% |

A 16% precision score means the agent is retrieving roughly 1 relevant snippet for every 6 it gets back. The agent always gets *some* results, but mostly noise. It's worth calling out that this metric is derived from LLM generated scores. It's useful, but should be taken with a grain of salt.

One likely cause is chunking strategy. Each chunk combines the document name, section header, and rule text. When the agent queries using the document name, it retrieves all rules from that document.  

---

## Conclusions and Next Steps

### What we learned

The hillclimbing process surfaced three distinct problem areas that needed investment:

1. **Iteration behavior** was a model capability problem. Swapping to a stronger model solved it. The agent went from single-query trajectories on 41% of failures to 6%.

2. **Retrieval accuracy** is a retrieval problem. Wrong filters account for ~50% of mismatches across every hard eval run. The agent is trying to apply rules, but not finding the right ones. Precision@5 at 16% confirms this.

3. **Analysis tooling** helps focus hillclimbing efforts around real problems. This may not be perfect, but is an effective way to quickly dive deep into categories of failures.

### What to do next

A few concrete directions:

- **Rethink chunking.** Currently, chunks combine document name + section header + rule text. This causes all rules from a document to match on document-name queries. Better approach: chunk at the rule level only, and index them without the document name leaking into the query surface. This should sharply improve precision.

- **Add a ranking function.** Before the agent runs, evaluate whether the retrieved rules are relevant to the question. This can be done with a cheap classifier and the signal can drive automatic re-querying or query rewriting.

- **Consider hybrid retrieval.** Lexical search is precise when it hits, but misses on vocabulary mismatch. A sparse + dense hybrid would give the coverage benefits of semantic search without sacrificing precision. Worth trying once the chunking issue is fixed.

- **Fix Agent max iterations.** AGENT_ERROR cases account for 19-23% of failures in later iterations. A quick glance shows these are cases where we reach max iterations. Likely low hanging fruit to fix.

- **Maintain the evaluation harness.** The LLM judge and behavioral analysis have been the most durable investment in this project. Further investments can be made to classify multiple source of failures per query. 

- **Encourage more reasoning.** Most rows follow the steps in the prompt exactly. It would be beneficial to introduce tools to help the model reason. This could literally be a reasoning tool, but a task list tool would also help. Then, the agent can reflect on search results and potentially re-search for more results.
