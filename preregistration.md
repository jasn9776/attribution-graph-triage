# Pre-registration: when should you not trust an attribution graph?

**Author:** Jason Soo
**Written:** [16-09-2026 6:17pm] — before any corpus data is collected
**Status:** append-only. Do not edit after timestamping; add a dated entry to the Deviations log instead.

---

## 0. What I already know, and how

Honesty requires separating what I am predicting from what I have already seen. Day 1 involved a
10-prompt pilot, so some predictions below are **data-informed**, not blind. Flagged where so.

Established on Day 1, on `google/gemma-2-2b` with GemmaScope per-layer transcoders via
`circuit-tracer`:

| Finding | Value |
|---|---|
| Replacement score, France prompt | 0.727 (library's own: 0.7196) |
| Completeness, France prompt | 0.936 (library's: 0.9268) |
| Run-to-run noise floor (bf16) | ~5e-05 on replacement score |
| Structure reproducibility | `n_nodes` identical across runs |
| Series termination | ~21 hops, confirming DAG structure |
| Node layout | errors then embeddings, contiguous; error grid is layer-major (confirmed by mod-6 zero pattern) |
| `max_feature_nodes=8192` artefact | saturation injected spurious error mass at the final token position; resolved at 16384 |
| Pilot replacement range (n=10) | 0.684 – 0.755 |
| Pilot completeness range (n=10) | 0.925 – 0.942 |

The ~0.008 discrepancy between my implementation and the library's is 150x the noise floor and
therefore a genuine methodological difference, not rounding. I use the library's implementation as
primary and mine as a cross-check. To be resolved by reading `METRIC_FN`'s source; recorded either
way.

---

## 1. Question

Anthropic report that attribution graphs gave satisfying insight on roughly a quarter of prompts
tried, and state this is difficult to quantify precisely — it is a self-report over an unspecified
prompt set, not a measurement. Each attempt costs hours of expert time. No pre-flight check exists.

The library computes replacement score and completeness, and the methods paper notes these are
useful "to flag prompts our method performs poorly on" — then never develops that use. The metrics
are reported as dataset-wide averages for comparing dictionaries, never as a per-prompt triage
instrument, and nobody has tested whether they predict a human's success at reading a graph.

**Primary question.** Across a stratified prompt corpus, do cheap prompt-side properties predict
attribution-graph quality well enough to serve as a triage rule?

**Secondary question.** Does replacement score track whether a human can actually extract a
mechanism from the graph?

---

## 2. Design

**Model:** `google/gemma-2-2b`, bf16, GemmaScope per-layer transcoders, nnsight backend,
`max_feature_nodes=16384`, `batch_size=32`. Full config in `day1_config.json`.

**Corpus:** ~300 prompts, 9 categories, ~33 each. Every category is grounded in a limitation the
methods paper documents qualitatively but never quantifies.

| # | Category | Grounded in |
|---|---|---|
| 1 | Factual recall | published success case |
| 2 | Two-digit arithmetic | published success case |
| 3 | Syntactic agreement | short mechanical control |
| 4 | Multi-hop | longer causal chains, error accumulation |
| 5 | Code completion | untested territory |
| 6 | Induction / copying | attention blind spot (frozen QK) |
| 7 | Multiple choice | attention blind spot |
| 8 | Known vs unknown entity | inactive-feature / suppression blind spot |
| 9 | Obfuscated text (4 corruption rungs) | reconstruction-error blind spot |

**Controls.** Prompt length varied deliberately *within* each category (short / medium / long) so
length is not confounded with category. Hard ceiling at `MAX_TOKENS`, applied identically to all
categories. Every prompt filtered to top-1 confidence in a specified band, so no category is
dominated by trivially certain or near-random continuations.

**Saturation guard.** Any graph reaching the feature cap raises an exception rather than being
recorded. Cap saturation silently injected fake error mass during Day 1 and would otherwise have
produced a spurious category effect, since longer prompts are unevenly distributed across
categories.

---

## 3. Outcomes

**Primary outcome:** replacement score (library implementation).

**Secondary outcomes:** completeness; total error mass; `error_by_layer` (26 values, comparable
across all prompts); `error_by_position` (length-varying, reduced to scalars — see 6.3); mean path
length; node and edge counts; pruning curve at thresholds 0.95 / 0.9 / 0.8 / 0.7.

**Predictors, all computable in milliseconds without running attribution:** category; token count;
next-token entropy; top-1 probability; mean token log-frequency (out-of-distribution proxy).

The predictors must be cheap. A triage rule requiring the attribution first is useless.

---

## 4. Predictions

### P1 — Category ranking (blind)

Predicted rank order by replacement score, best to worst:

1. Factual recall
2. Syntactic agreement
3. Two-digit arithmetic
4. Known vs unknown entity
5. Multi-hop
6. Code completion
7. Multiple choice
8. Induction / copying
9. Obfuscated text

Scored by Spearman correlation between predicted and observed ranks.

### P2 — Replacement score has low between-category variance (data-informed)

**I predict replacement score will not discriminate between categories.** Specifically: the
between-category range of category means will be **under 0.15**, and category will explain
**under 20%** of variance in replacement score.

*This is informed by pilot data.* Ten prompts spanning factual recall, arithmetic, syntax,
multi-hop, code and induction produced a replacement range of 0.684–0.755 — a spread of 0.071
against a noise floor of 5e-05. The differences are real but small. I am extrapolating that
tightness to the full corpus, including three categories the pilot did not cover (multiple choice,
known/unknown entity, obfuscated text).

**The extrapolation is the risky part.** Obfuscated text in particular is the case where the
methods paper reports near-total error-node dominance, so it could break the pattern badly. If it
does, P2 fails and that is the more interesting outcome.

### P3 — Error location has high between-category variance (data-informed direction, blind magnitude)

**I predict error *location* will discriminate where error *quantity* does not.** Specifically,
category will explain **more variance in error-location summaries than in replacement score**,
by a factor of at least 2.

Mechanistically: obfuscated text should concentrate error on corrupted tokens in early layers;
induction may concentrate at the copy-source position; factual recall should be low and flat.

### P4 — Completeness is even less discriminative than replacement

Pilot completeness ranged 0.925–0.942, a spread of 0.017 against replacement's 0.071. I predict
completeness will have a smaller between-category range than replacement across the full corpus.

### P5 — Human ratings (blind)

Spearman correlation between replacement score and my 3-point "did I learn a mechanism" rating on
40 graphs will be **positive but weak: |rho| < 0.4**.

Rationale: if P2 holds and replacement barely varies, it cannot track something as coarse as
whether a human understood the graph.

---

## 5. Falsification

The project has a null result if:

- Category explains **under 10%** of variance in every outcome, and no predictor reaches
  significance after Benjamini-Hochberg correction. Then no triage rule exists and I say so.
- Leave-one-category-out cross-validated R² is **at or below zero** for all outcomes. Then any
  in-sample structure does not generalise to unseen prompt types, which is the actual use case.

**I will report a null result as the headline finding rather than reframing around a surviving
sub-analysis.** A documented "these metrics do not support triage" is worth more to the field than
a fished positive.

Specific ways I could be wrong, stated now:

- If P2 fails and replacement *does* discriminate strongly, my central claim inverts and the metric
  is more useful than I expect. Good outcome, different paper.
- If P3 fails and error location is also flat, the project has no signal in either variable and
  the honest conclusion is that per-prompt triage is not achievable with these instruments on this
  model.
- If human ratings are self-inconsistent (intra-rater agreement below 0.6), P5 is uninterpretable
  and I report that instead of the correlation.

---

## 6. Analysis plan

### 6.1 Models

Outcomes are proportions bounded in (0,1). Beta regression, or logit-transform then OLS. Not plain
linear regression on a proportion.

- Outcome ~ category + token_count + next_token_entropy + top1_prob + mean_token_logfreq
- Effect sizes with confidence intervals, not p-values alone
- Benjamini-Hochberg across the 9 category contrasts and across outcomes

### 6.2 Cross-validation

Two schemes, both reported:

- **Held-out prompts within known categories.** The easy version.
- **Leave-one-category-out.** The version that matters, because the real use case is a prompt type
  I never tested. I expect this to be substantially worse and will say so plainly.

### 6.3 Handling variable-length error profiles

`error_by_position` has one value per token, so lengths differ across the corpus and the arrays
cannot be stacked directly. Pre-specified reduction to scalars:

- fraction of error mass in the first third / middle third / final third of positions
- position index of maximum error, normalised to [0, 1]
- Gini coefficient of the position profile (concentration vs spread)
- error mass at the second-to-last position

`error_by_layer` is always 26 values and needs no reduction. Summaries: fraction in the first
third / last third of layers, and layer index of maximum error.

**Note:** error mass at the *final* token position is structurally 0.000 once the feature cap is
not binding, so it is excluded as an outcome. This is itself a limitation of the instrument and
will be reported: the method's blind spots at the read-out position are not measurable this way.

### 6.4 Noise floor

Any claimed effect must exceed 5e-05 by a wide margin. Observed pilot spread (0.071) is ~1400x the
floor, so real effects are detectable; the question is whether they are *large*, not whether they
are *nonzero*.

### 6.5 Human validation

40 graphs sampled to span the observed range of replacement score, not stratified by category —
spread on the validated variable is what matters. Rated blind to all metrics, randomised order,
5-minute cap per graph, 3-point scale. 10 unmarked duplicates for intra-rater reliability, reported
regardless of outcome. Single-rater and non-blind-to-hypothesis; both stated as limitations.

---

## 7. Scope limits, stated in advance

- One model, one size, one dictionary family. Everything may differ at 9B or with cross-layer
  transcoders.
- bf16, not fp32. T4, not A100.
- Metrics measure whether a graph **covers** the computation, not whether it covers it
  **correctly**. Mechanistic faithfulness requires per-graph intervention experiments and is out
  of scope. This is the deepest limitation and will be stated prominently.
- Prompt categories are hand-defined by me and not exhaustive.
- I formed P2–P4 after seeing 10 pilot prompts.

---

## 8. Deliverables, committed regardless of outcome

1. Prompt corpus with per-prompt metrics, released on HuggingFace
2. The regression, with honest cross-validated performance
3. A one-page triage card, or an explicit statement that no usable rule was found
4. Write-up with epistemic status up front, posted to LessWrong / Alignment Forum

---

## 9. Deviations log

Append dated entries. Do not edit anything above.

| Date | Deviation | Reason |
|---|---|---|
| | | |

### Deviation, [17-09-2926] — filter switched from top-1 probability to next-token entropy

On measuring the corpus, top-1 probability was found to capture *continuation*
confidence rather than *answer* confidence on base-model prompts. The model completes
"The capital of France is" with " a" (p=0.223), continuing the sentence rather than
answering the implied question, and completes "{name} was born in the year" with a
leading space (p≈0.84) for both known and unknown entities, since the discriminating
token is one position further on.

The filter's purpose is narrow: exclude prompts where the model is near-deterministic
(no computation left to explain) or near-uniform (no coherent mechanism to find).
Next-token entropy serves that purpose more robustly, since it uses the full
distribution rather than a single point and is not distorted when mass splits across
near-identical tokens.

Entropy does not solve the answer-versus-continuation problem, and is not claimed to.
Both top-1 probability and entropy are retained in the corpus as covariates, and no
analysis conclusion should depend on which was used for exclusion. The band is chosen
from the measured per-category distribution, and if no band retains every category
above n=20 the filter is dropped entirely in favour of covariate adjustment.

### Deviation, [17-09-2926 4:43PM] — Length Balance Note
Length balance. Final eta-squared (category → n_tok) = 0.204 on 281 prompts. Eight of nine categories fall within ±1.8 tokens of the grand mean; multiple choice runs +4.8 by construction, as the format requires a stem plus two options. MCQ was already compressed from three options to two to reduce this. Category effects are reported controlling for n_tok, and the MCQ contrast specifically is reported with that caveat noted.

### Deviation, [17-09-2926 4:43PM] — Obfuscation Prompts/sentences removed
Obfuscated ladder. Four of twelve base sentences dropped because character substitution inflated their token count beyond MAX_TOKENS; eight complete four-rung ladders retained (32 prompts). Corruption level and token count are intrinsically confounded within this category — substituted text fragments into character-level tokens, which is the mechanism being tested — so the confound is reported rather than controlled.
