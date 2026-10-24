# RAG comparison

Baseline is the preserved `evals/results.json` PDF-only run.
Improved is `evals/results_improved.json` when that file exists.
Agentic results are in `evals/results_agentic.json`.
Gold answers were not passed to answer generation.

## Full improved run

Metrics below for the improved system use every saved question, not only the sample.

| System | Questions | Correct | Grounded | Page hit | Abstention on unanswerable | Latency and usage |
|---|---|---|---|---|---|---|
| Baseline PDF-only, all saved questions | 39 | 39/39 (100.0%) | 39/39 (100.0%) | 23/24 (95.8%) | 3/3 (100.0%) | 6.95s | errors 0, unscored 0, tokens unavailable |
| Improved PDF RAG, all saved questions | 39 | 38/39 (97.4%) | 38/39 (97.4%) | 23/24 (95.8%) | 2/3 (66.7%) | 9.50s | errors 0, unscored 0, tokens unavailable |

## Representative questions

Questions 1, 3, 14, 25, 30, 39.

| System | Questions | Correct | Grounded | Page hit | Abstention on unanswerable | Latency and usage |
|---|---|---|---|---|---|---|
| Baseline PDF-only | 6 | 6/6 (100.0%) | 6/6 (100.0%) | 2/3 (66.7%) | 2/2 (100.0%) | 7.90s | errors 0, unscored 0, tokens unavailable |
| Improved PDF RAG | 6 | 6/6 (100.0%) | 6/6 (100.0%) | 2/3 (66.7%) | 1/2 (50.0%) | 7.77s | errors 0, unscored 0, tokens unavailable |
| Agentic RAG | 6 | 6/6 (100.0%) | 6/6 (100.0%) | 2/3 (66.7%) | 2/2 (100.0%) | 6.74s | errors 0, unscored 0, tokens 37141 |

## External search

External search was not run. Set TAVILY_API_KEY or BRAVE_SEARCH_API_KEY in .env to enable web search. Agentic answers used PDF evidence only and could abstain.

API dollar cost was not returned by Gemini. Token totals are included only when the API response had usage metadata.

The agentic judge saw the reference answer only after the model answer was generated.

The full improved run's only miss is question 5. The vision read of page 5 changed `t1` and `t4` into `f1` and `tu`, so correctness and groundedness are both false. Questions 1, 3, 14, 25, 30, and 39 do not include question 5, which is why the six-question improved sample is 6/6.

Improved abstention is 2/3 in the table because the saved keyword flag did not treat "no information" as a refusal. The judge still marked question 14 as a correct refusal. Baseline abstention was recomputed from answer text and did match all three refusals. That difference is the detector, not a new invented answer.

Improved-run token counts were not captured: the evaluator copied the usage field before the API response updated it. That is fixed for later runs. The agentic total of 37141 tokens is from the responses that included usage metadata. No dollar cost was returned.

- Q1 ANSWERED in 8.39s. Web results: 0. 
- Q3 ANSWERED in 14.65s. Web results: 0. 
- Q14 ABSTAIN in 3.16s. Web results: 0. 
- Q25 ANSWERED in 8.18s. Web results: 0. 
- Q30 ANSWERED in 3.28s. Web results: 0. 
- Q39 ABSTAIN in 2.79s. Web results: 0. 
