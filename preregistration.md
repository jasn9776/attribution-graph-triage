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

### Deviation, [17-09-2926 5:13PM] — Length Balance Note UPDATED
Length balance. Eta-squared (category → token count) = 0.221 across all nine categories, but 0.081 with multiple choice excluded. Eight of nine categories fall within ±1.8 tokens of the grand mean; multiple choice runs +4.7 by construction, since the format requires a stem, two options and an answer cue. The category was already compressed from three options to two to reduce this. Category effects are reported controlling for token count, and the multiple-choice contrast is flagged specifically.

### Deviation, [17-09-2926 5:20PM] — Collinearity between category and entropy.
Collinearity between category and entropy. Next-token entropy is strongly determined by category (eta-squared = 0.720), ranging from 0.67 for multiple choice to 4.16 for obfuscated text. This is a genuine property of the domain rather than a design fault: prompt types differ in how much uncertainty they leave the model in. Entropy and token count are close to independent (r = −0.194), so the continuous predictors are not themselves collinear.

The two predictors are therefore not entered into a single model as if independent. Three models are reported: (a) category alone, (b) cheap continuous predictors alone — entropy, top-1 probability, token count, token-frequency proxy — and (c) both, with variance inflation factors. If VIF exceeds 5 the combined model is reported for completeness but not used for inference.

**Model (b) answers the question that matters practically.** A practitioner with a prompt has no category label; they have the prompt. If cheap continuous predictors alone predict graph quality under leave-one-category-out cross-validation, that is a usable triage rule regardless of whether the effect can be attributed to category or to entropy.

### day1_final notebook versions for packages and config (run on Kaggle notebook)  [21-09-2026 2:10PM]
torch              2.10.0+cu128
transformers       4.57.3
transformer_lens   3.2.1
circuit_tracer     0.5.0
nnsight            0.7.0
numpy              2.0.2
scipy              1.16.3
gpu                Tesla T4

### Feature Cap raised to 32768 [21-09-2026 3:12PM]
**Feature cap raised to 32768.** At 24576 the densest pilot prompt (induction, nonce bigrams) had 1,700 features of headroom. Nonce text fragments into many subword tokens and is the most feature-dense input in the corpus. Attribution time was measured to be independent of the cap (13.3s vs 13.4s at 16384 vs 24576), so raising it costs nothing and prevents induction prompts being rejected by the saturation guard.

## SUMMARY of changes after debugging day1_final with claude [21-09-2026 3:23PM]
Position summaries on interior positions only.\
excluding BOS and the final token, because error there is structurally zero once the cap does not bind. Including them would inflate the Gini coefficient for reasons unrelated to the prompt.\
Feature cap raised to 32768\
Pruning curve replaced by node-count curve, library version on a 30-graph subsample\
Metrics on GPU in float32, validated against float64 to 3×10⁻⁸

Full version of changes:
## Day 1 final — changes and deviations
**21 September 2026, 15:23**

Four changes made while finalising the Day 1 pipeline. Each alters how an outcome is
measured, so each is recorded here rather than made silently.

### 1. Position summaries computed on interior positions only

**Change.** Position thirds, argmax and Gini are computed on positions 1 to n−2,
excluding the BOS token and the final token.

**Why.** Error at those two positions is structurally zero whenever the feature cap is
not binding — a property of how circuit-tracer builds the graph, not of the prompt.
Treating these guaranteed zeros as data inflates concentration measures, and by an
amount that depends on prompt length: two zeros are a third of a 6-token profile but a
tenth of a 20-token one. That would reintroduce the length confound the corpus design
controls for. On a 6-token example, Gini falls from 0.49 to 0.23 when the structural
zeros are excluded.

**Evidence.** BOS and final-position error were exactly zero on all 12 pilot graphs,
across all nine categories. Both are recorded for every corpus graph and verified to be
zero before analysis.

### 2. Feature cap raised from 24576 to 32768

**Why.** Nonce text is the most feature-dense input in the corpus, because nonsense
words fragment into many subword tokens. The densest pilot prompt (induction) had only
1,700 features of headroom at 24576. A saturating graph is rejected by the saturation
guard, so the risk was losing prompts from the category most likely to thin out.

**Cost.** None measurable. Attribution time is independent of the cap (13.3 s vs 13.4 s
at 16384 vs 24576); a graph only uses the features it needs.

### 3. Pruning curve replaced by a node-count curve

**Change.** Each corpus graph records the number of nodes needed to reach 95 / 90 / 80 /
70 % of total logit influence. The library's `prune_graph` curve is computed on a
stratified subsample of 30 graphs to measure agreement between the two.

**Why.** `prune_graph` at four thresholds took 115 s of a 172 s per-graph total on a
17-token prompt (67 %). The node-count curve is the node-threshold step of that
algorithm, without edge pruning or orphan removal — a proxy, and reported as one.

### 4. Metrics computed on GPU in float32

**Change.** `graph_metrics_fast` replaces the float64 NumPy implementation for the
corpus run. The NumPy version is retained as a reference.

**Validation.** The two agree to within 3.45 × 10⁻⁸ on every reported quantity, on two
graphs including the largest in the pilot — about three orders of magnitude below the
run-to-run noise floor of 5 × 10⁻⁵. Node, feature and token counts agree exactly.

**Effect.** Metric time fell from 15.3 s to 0.5 s per graph. Attribution is now ~99 %
of runtime: 24.0 s mean per graph, so ~1.9 h for 280 graphs (2.8 h with margin).

### Also noted

Peak GPU memory reached 12.3 GB of 14.56 on the densest pilot graph, since the fast path
holds the full adjacency on the device. Day 3 retries any out-of-memory graph on CPU
rather than dropping it.

## Position summaries and prompt length [21-09-26 3:26]
Position summaries and prompt length. Excluding BOS and final positions leaves n−2 interior positions, as few as 3. Raw Gini has a maximum of (n−1)/n, so it scales with length independently of the prompt. Position concentration is analysed using normalised Gini, G × n/(n−1), computed from the stored per-position profiles. Prompts with fewer than 5 interior positions ([k] of them) are retained, but position summaries are reported both with and without them.

**Length balance.** Final eta² (category → token count) = 0.131 across 8 categories and 255 prompts. The largest deviations from the grand mean are induction (+2.5 tokens) and entity (−1.6). Multiple choice, previously the main source of length imbalance (overall eta² 0.253 with it included), was dropped for failing its manipulation check. Token count is carried as a covariate.


## Corrections to earlier entries [22-09-2026 3:26]

This log is append-only, so earlier entries are corrected here rather than edited.

Date typo. Four entries are dated "17-09-2926". The year should read 2026. They were written on 17 September 2026.
Placeholder filled. In "Position summaries and prompt length", "[k] of them" is 27: code completion 4, obfuscated 14, syntactic agreement 9.
Superseded: obfuscation entry (17-09, 4:43PM). Stated 8 complete ladders / 32 prompts. Final is 11 ladders / 44 prompts — see D2.7.
Superseded: collinearity entry (17-09, 5:20PM). Stated entropy eta² = 0.720 and r = −0.194, with multiple choice as the lowest-entropy category. Final values, on the 8-category corpus, are eta² = 0.753 and r = −0.109. The three-model analysis plan in that entry is unchanged.
Superseded: both length-balance entries (17-09). Final length eta² is 0.131, see D2.11.
Superseded: Section 2 configuration. max_feature_nodes=16384 and the confidence filter are both replaced — see "Final configuration as of Day 3" at the end of this log.

**Day 2 — corpus construction [22-09-2026 3:26]**

Every change below was made on the basis of manipulation checks, which test whether a prompt engages the mechanism its category is named for. None was made on the basis of attribution-graph outcomes, which had not been collected for any corpus prompt at the time of writing.

#### D2.1 Induction rebuilt

Sentence-frame prompts ("The {nonce} ran fast. The {nonce}") did not test induction: the model predicted ran or was in 20/20 cases, completing the sentence frame rather than copying the nonce token. They were dropped.

The category now uses bigram repetition only, "a b a b a", with each nonce pair in both orders (24 prompts). Copying is 24/24 by first-token prefix match, and 24/24 under a strict test requiring the first three characters over a three-token greedy continuation. Six first-token predictions were single characters; all six continued into the correct word.

Two earlier implementations of the copying check were wrong in opposite directions — one counted any predicted token appearing in the prompt (reported 88%), one required an exact whole-word match despite nonce words tokenising into fragments (reported 0%). The prefix-match version is used.

#### D2.2 Multi-hop cities corrected

Five of the original ten cities — Phoenix, Atlanta, Boston, Denver, Nashville — are their own state capital, so the answer was present in the prompt and could be copied without a second hop. Replaced with non-capital cities: Dallas, Seattle, Miami, Detroit, Chicago, Buffalo, Pittsburgh, Milwaukee, Memphis, Omaha.

#### D2.3 Fact: prefix adopted for short answer frames

The attribution graph explains whatever the model predicts, so a factual prompt that does not elicit the answer produces a graph of something else. Answer rates were measured with and without a Fact: prefix (the framing used in circuit-tracer's own demo):

subtype	bare	with Fact:
capital_short	0.33	0.92
state_capital	0.80	1.00
sport_short	0.75	0.75
city_language_long	0.80	0.60
state_capital_long	1.00	0.80
overall	0.72	0.89

The prefix is applied to the short frames only (capital_short, sport_short, state_capital), because it lowers the answer rate on two long frames. Short and long frames within factual recall and multi-hop therefore differ in format as well as length; subtype records this. Final task success with the prefix: factual recall 0.80, multi-hop 0.94.

#### D2.4 Multiple choice dropped after two failed designs

Design 1, letter options. Gemma-2-2B (base) does not perform multiple-choice symbol binding. The arrow format (... -> () continued the option list: 12/12 predicted c, an option that does not exist. The answer format (... Answer: () defaulted to a, chosen 75% of the time; accuracy was 58%, against 42% for always answering a. Constrained choice between the two letter logits was also 58%, so the failure is in binding answers to letters, not in the top-1 token masking a known answer.

Design 2, forced choice between option words, "{stem} {x} or {y}? {stem}", each item in both orders. Constrained accuracy 71%, top-1 71%, first-listed option chosen 79% of the time — failing the criteria set before the check was run (≥80%, ≥70%, 30–70%). All seven errors chose the first-listed option; accuracy was 83% when the correct option came first and 58% when it came second. The cause is the format: repeating the stem creates an induction pattern, and the model copies whatever followed the stem's first occurrence, overriding factual knowledge.

Per the rule stated with the check, the category was dropped rather than redesigned a third time, since a third format chosen after two observed failures would amount to selecting a format by its result. The attention blind spot remains covered by induction.

The design-2 failure is itself a small observation — in-context copying overriding stored factual knowledge — and will be reported descriptively. It is not part of the confirmatory analysis. The check output is saved as check_forced_choice.csv from the run of 22-09-2026.

#### D2.5 Known vs unknown entity: construction and manipulation check

16 matched pairs. Each pair shares a first name and has identical token count within the frame "{name} was born in the year", measured in the frame rather than standalone, since famous surnames get their own token precisely because they are frequent. Each unknown name uses a distinct invented surname, so no single invented token dominates the unknown arm.

Manipulation check. Confidence does not separate the arms: mean top-1 probability 0.819 known vs 0.837 unknown, because both predict a leading space before the year. The generated year does: greedy 6-token continuations give round-numbered years (ending 0 or 5) for 5/16 known vs 16/16 unknown, Fisher exact p = 6.77 × 10⁻⁵. Known names produce specific years, spot-checked against reality; all five round known years are correct years that happen to end in 0 or 5. The model is not uncertain about fabricated people — it is confidently wrong.

Composition. 13 of 16 pairs use women's names, because less-tokenised surnames correlate with lesser fame, which correlates with historical under-recognition. Not expected to affect the measurement; recorded so the composition is not mistaken for a choice.

The long frame ("One fact widely reported about {name} is that they were") was dropped: it predicted the at p ≈ 0.1–0.27 for both arms and measured nothing.

#### D2.6 Syntactic agreement validated

The 18 noun-phrase prompts pair singular heads with plural distractors and vice versa (agreement attraction). Where the model predicts a number-marked verb (10/18), it agrees with the true head in 10/10. Comparing plural and singular verb logits directly (were−was + are−is), which covers all 18, agreement is 18/18, with mean margin +12.2 for plural heads and −7.5 for singular heads, every item on the correct side. No attraction errors.

The remaining 8 NP prompts continue the noun phrase (of, in) or predict number-neutral had. The 8 fill_ prompts ("The cat sat on the") involve no agreement and serve as a syntax control.

An earlier run reported 0% agreement because a pandas column named head collided with the DataFrame.head() method. Corrected before any conclusion was drawn.

#### D2.7 Obfuscation ladder, final

12 base sentences × 4 rungs (clean, typo, random capitals, character substitution). One base (b11) was dropped because its substitution rung reached 21 tokens; only complete four-rung ladders are kept, so every rung comparison is within the same sentence. Final: 11 ladders, 44 prompts.

A bug in which the typo rung could select a word of three letters or fewer and change nothing was fixed; the generator now asserts no typo rung equals its clean base.

Next-token entropy by rung is 3.68, 4.19, 4.96, 4.28 — not monotonic in corruption severity, replicating the pattern seen on different corrupted text on 17-09. Token count by rung is 5.6, 6.8, 11.0, 16.0: corruption and length are intrinsically confounded, because substituted text fragments into character-level tokens, which is the mechanism under test. Reported, not controlled.

#### D2.8 Confidence filter dropped

The 17-09 entry specified that the filter would be dropped if no entropy band retained every category above n = 20. None did — the gentlest band tested reduced the smallest category to 18. The filter was therefore dropped. Next-token entropy and top-1 probability are carried as covariates. A filter would have imposed different selection pressure on different categories; a covariate does not.

#### D2.9 Task success recorded per prompt

task_ok is stored for every prompt with a single correct answer:

category	n defined	rate
arithmetic	28	0.96
factual recall	40	0.80
multi-hop	32	0.94
induction	24	1.00
syntactic agreement (NP prompts)	18	1.00

Left empty for code completion, entity, obfuscated text and syntax fill prompts, which have no single correct answer. Use is specified in D3.9.

#### D2.10 Two smaller corrections
Syntax prompts: "...that morning, The keys..." had a capital mid-sentence. Lower-cased.
Code completion: "raise ValueError(" fell below MIN_TOKENS (4 tokens) and was dropped.
D2.11 Final corpus

255 prompts, 8 categories.

category	n
obfuscated	44
factual recall	40
entity known/unknown	32
multi-hop	32
code completion	29
arithmetic	28
syntactic agreement	26
induction	24

Length eta² (category → token count) = 0.131. Largest deviations from the grand mean: induction +2.5 tokens, entity −1.6. Entropy eta² = 0.753; r(entropy, token count) = −0.109.

### Day 3 — decisions fixed before any corpus graph is attributed [22-09-2026 3:26]
#### D3.1 Which implementation is the primary outcome

Section 3 names the library's replacement score as primary. The corpus pipeline uses graph_metrics_fast (own implementation), which differs from the library by ~0.008 on the reference prompt — 150× the noise floor.

Rule, applied before the loop starts: time the library's metric function on the reference graph.

If it takes under 5 s per graph, it is computed for every corpus graph and remains primary. The own implementation is recorded alongside, and agreement is reported.
If it takes 5 s or more, the own implementation becomes primary, the library metric is computed on the 32-graph subsample in D3.5, and the correlation between the two is reported.

Measured: METRIC_FN = [__] s on the reference graph → primary = [library / own]. (**Ran at start of this in DAY3 notebook on HH:MM 2X-09-2026**])

#### D3.2 The 0.008 discrepancy

Section 0 committed to explaining the gap by reading the library's source.

Resolution: [state the difference — e.g. how logit nodes are weighted or how rows are normalised — or: "Not resolved. The library source was read on 21-09 but the difference could not be attributed to a specific step. Both values are recorded for every graph and the conclusions are checked under each."]

#### D3.3 Failure handling in the corpus run

Every prompt produces a row. No prompt is silently dropped.

Feature-cap saturation (graph rejected by the guard): recorded as status = saturated.
Out of GPU memory during metrics: retried once on CPU. If that succeeds, recorded normally with metrics_device = cpu; if not, status = oom.
Any other exception: status = error, with the message.
Non-zero BOS or final-position error (> 1 × 10⁻⁶): the graph is recorded but flagged status = structural_zero_violated and excluded from all position summaries, since the interior-position rule depends on those zeros.

Failures are reported as counts by category. Any category losing more than 10% of its prompts is flagged in the results, and its estimates are reported with that caveat.

#### D3.4 Model-consistency check

Day 2's cheap predictors, task_ok, and the Fact: decision were computed on the eager-attention HuggingFace model. Attribution runs on the nnsight ReplacementModel. Before the loop, 24 corpus prompts are sampled — 3 per category, random_state = 0 — and top-1 tokens compared between the two models.

Action, fixed now: if top-1 disagrees on 2 or more of the 24, all cheap predictors (top1_prob, top1_token, next_token_entropy) and task_ok are recomputed on the ReplacementModel for every prompt, and those values are used in the analysis. If 0 or 1 disagree, the Day 2 values are used and the check is reported.

**D3.4 — outcome [22-09-2026 5:10PM].** 0/24 disagreements on exact decoded string (0/24 ignoring whitespace). Not triggered; Day 2 values for top-1, entropy and task_ok are used throughout. The three entity prompts agree on a leading space before the year, a weaker test than the 21 prompts agreeing on content tokens. Top-1 was read from the graph's logit_tokens for every prompt without failure. Whole-corpus agreement is reported after the run.

#### D3.5 Pruning-curve subsample

The library's full prune_graph curve is computed on 32 graphs: 4 per category, sampled with random_state = 0 from successfully attributed prompts. Agreement with the per-graph node-count curve is reported as a correlation at each threshold.

#### D3.6 P1 with multiple choice removed

P1 is scored on the 8 remaining categories, with multiple choice deleted from the predicted ranking and the relative order of the others unchanged:

Factual recall · 2. Syntactic agreement · 3. Two-digit arithmetic · 4. Known vs unknown entity ·
Multi-hop · 6. Code completion · 7. Induction · 8. Obfuscated text
#### D3.7 P3 primary measure

P3 compares variance explained by category in error location against replacement score. Section 6.3 lists seven error-location summaries; testing all seven against one threshold would make a pass likely by chance.

Primary: normalised interior Gini of the per-position error profile, G × n/(n−1) on positions 1 to n−2. Test: eta² of category on normalised Gini is at least 2× eta² of category on logit-transformed replacement score. The other six summaries are reported as secondary and do not count toward P3.

#### D3.8 Human rating protocol

Rubric, fixed before any graph is viewed:

2 — I can state a causal path from input tokens to the output in one sentence, naming the intermediate features.
1 — I can identify at least one meaningful intermediate feature, but cannot trace a complete path.
0 — Neither.

Sampling. 40 graphs: replacement scores of all successful graphs are divided into 8 quantile bins, and 5 graphs drawn per bin with random_state = 0. 10 of the 40 are then re-inserted as unmarked duplicates, giving 50 presentations in randomised order.

Presentation. Pruned at node threshold 0.8, viewed in the circuit-tracer graph viewer with all metric values hidden. 5-minute cap per graph.

Reliability. Intra-rater agreement on the 10 duplicates is reported as weighted kappa. If below 0.6, P5 is reported as uninterpretable (Section 5).

#### D3.9 Multiple comparisons and the task-success subset
Benjamini–Hochberg is applied across the 8 category contrasts (not 9) and across outcomes.
Every category effect is reported twice: on all prompts (primary), and on the task_ok = True subset where defined (secondary). The subset analysis tests whether category effects are driven by prompts where the model did not do the intended task.
Final configuration as of Day 3 [22-09-2026 HH:MM] **set after confirmation**
setting	value
Model	google/gemma-2-2b, bfloat16
Transcoders	GemmaScope per-layer (gemma)
Backend	nnsight
max_feature_nodes	32768
batch_size	32
lazy_encoder	True
Metrics	graph_metrics_fast, GPU float32 (agrees with float64 to 3 × 10⁻⁸)
Pruning	node-count curve per graph; library prune_graph on 32-graph subsample
Position summaries	interior positions only; normalised Gini
Noise floor	~5 × 10⁻⁵ on replacement score
Corpus	255 prompts, 8 categories
MAX_TOKENS / MIN_TOKENS	20 / 5
FACT_PREFIX	"Fact: ", short frames only
Confidence filter	none; entropy and top-1 as covariates
Hardware	Kaggle T4
Packages	torch 2.10.0, transformers 4.57.3, circuit-tracer 0.5.0, nnsight 0.7.0
Expected runtime	~24 s per graph, ~1.7 h for the corpus

## Adjustment to D3.1.
 After the correction in D3.2, graph_metrics_fast reproduces the library's compute_graph_scores to within [δ] on the reference graph and on [one long prompt]. The library's definition therefore remains the primary outcome, computed via the fast path. compute_graph_scores is also called on the 32-graph subsample in D3.5 as an ongoing check.

## D3.2 — resolved.
The 0.008 gap was a bug in my implementation, not a difference of method. The library locates logit nodes by position — the last n_logits nodes, in the order of logit_probabilities. My implementation located them structurally, as nodes with no outgoing edges, which also captures dead features; narrowing that set by incoming weight reordered the logits, so probabilities were assigned to the wrong nodes. Completeness also differed by definition: the library averages the non-error input fraction over all nodes, weighted by influence plus the logit probabilities. Both corrected on 22-09-2026 by adopting the library's layout and definition. The library's layout independently confirms the errors-first ordering established from graph structure on Day 1.
### Consequence
The Day 1 pilot values in Section 0 were computed with the implementation corrected in D3.2 and are biased by roughly 0.008. Predictions P2 and P4 were informed by the spread of those values (0.071 and 0.017), which the bias does not materially change. P2 and P4 are retained as written.

## D3.10 — outcome, [22-09-2026 4:57PM]. 
Branch 2. On four pilot prompts under the corrected implementation, BOS error is 0.000000 on all four; final-position error is 0.044–0.167, exceeding the largest interior position on every prompt (on the France prompt, about 60% of total error). Position summaries use positions 1 to n−1; D3.3's structural-zero flag applies to BOS only; n_interior is recomputed as n−1 at analysis (the saved corpus.csv column uses n−2). Layer-major reshape re-confirmed: [26 zeros, all at residue 0 mod n_tok; flat under mod 26].

This reverses the note in Section 6.3. Error at the read-out position is measurable and is the dominant error location on all four pilot prompts. A secondary summary, err_final_share, is recorded per graph. Normalised interior Gini remains the P3 primary measure as fixed in D3.7; because the final position dominates, it is expected to largely reflect the final-position share, which will be reported alongside it.

**Result:** all 30 zeros match the layer-major prediction exactly — 26 at BOS (every layer) and 4 at the last layer's positions 1 to n−2. No unexplained zeros; no predicted zeros missing. Position-major does not match.


### D3.4 — refinement to the model-consistency action [22-09-2026 04:59]

**Written before the check was run.** Refines the action pre-registered in D3.4. The check itself —
24 prompts, 3 per category, `random_state = 0`, threshold of 2 or more disagreements — is unchanged.

**What is compared.** The top-1 token from Day 2's eager-attention HuggingFace model (`top1_token`
in `corpus.csv`) against the ReplacementModel's top-1, read from the attribution graph's logit
nodes as the highest-probability logit. Agreement is judged on the **exact decoded string**. A
whitespace-insensitive comparison is also reported but does not count toward the threshold.

**If the ReplacementModel's top-1 cannot be read from a graph**, that prompt counts as a
disagreement. If it cannot be read for most prompts, the extraction is treated as broken: it is
repaired and the check rerun before the corpus loop, and the failed run does not count as a
trigger.

**Refined action if triggered (2 or more disagreements):**

- **Top-1 token and probability:** replaced by the ReplacementModel's values, for every prompt.
  These are recorded from each graph during the corpus run, so the replacement is exact and costs
  nothing.
- **`task_ok`, recomputed from the ReplacementModel's top-1** for factual recall, multi-hop,
  arithmetic and induction, by prefix match against the expected answer. For induction this is the
  first-token prefix test, not the three-token strict test used on Day 2, since a three-token
  continuation cannot be recovered from the graph's logits.
- **`task_ok` for syntactic agreement: retained from Day 2.** It is defined by the plural-minus-
  singular verb-logit margin, and the graph stores only the top logits, which need not include
  both verb forms.
- **Next-token entropy: retained from Day 2.** The graph stores only the top logits, so the full
  distribution cannot be recovered. Entropy also serves as a *cheap predictor* — something a
  practitioner computes on a standard model before attributing anything — so the standard
  eager-attention model is the appropriate source regardless.

**Reason for the refinement.** The original action said to recompute *all* cheap predictors on the
ReplacementModel. Doing so for entropy would require a separate full forward pass through the
attribution wrapper, which has not been validated for that use under the nnsight backend. The
refinement recomputes exactly what the graphs make available and keeps the rest on the model a
practitioner would use.

### Corpus run — outcome [22-09-2026 6:45PM].
255/255 prompts attributed with status ok. No feature-cap saturation, no out-of-memory failures, no BOS structural-zero violations, no metrics retried on CPU; no category lost prompts. Mean attribution 18.1 s (max 53.9 s), 72 minutes for the loop; peak GPU 12.5 GB. Whole-corpus top-1 agreement between the Day 2 HF model and the ReplacementModel: 97.6% (6 of 255 differ). D3.4 was not triggered on the pre-registered 24-prompt sample, so Day 2 values are used; the 6 disagreements are listed in [file] and reported as a sensitivity check.

**Whole-corpus top-1 disagreements.**
6 of 255 (2.4%): p0008, p0013, p0087, p0117, p0219, p0237. Four do not affect task_ok: two are near-ties between plausible content words (glass/bowl, post/letter), one is a tokenisation difference (c/camera), and in one both models miss the answer (p0013, the/also). In two, the ReplacementModel produces the correct answer where the HF model produced a function word — p0008 (also → Rome) and p0117 (with → Italian) — so task_ok would flip from False to True. Per D3.4 (not triggered), Day 2 values are retained. The D3.9 task_ok subset analysis is reported both as registered and with these two prompts flipped. Both flips run in the same direction, towards the answer, which may reflect bf16 near-ties or a small backend difference in attention; two cases are too few to distinguish.

**If not triggered (0 or 1 disagreements):** Day 2 values are used throughout.

**Reported either way:** the 24-prompt result, and top-1 agreement between the two models across
the whole corpus, computed after the run from the per-prompt ReplacementModel values.

**D3.5 — outcome. 22-09-2026 7:36PM].**
Library pruning curve computed on 32 graphs (4 per category, random_state = 0). Spearman correlation between the per-graph node-count curve and the library's prune_graph curve: 1.000, 1.000, 1.000, 0.994 at thresholds 0.95, 0.9, 0.8, 0.7 (n = 32 each). The node-count curve is used for all corpus graphs. Maximum absolute difference from the library, including re-attribution: completeness 9.9×10⁻⁵, replacement 3.4×10⁻⁴ — the latter above the Day 1 noise floor of 5×10⁻⁵. Diagnosis: [same-graph difference ___; re-attribution difference ___ on prompt ___]. [Conclusion: re-attribution noise scales with graph size; noise floor revised to ___ for large graphs. / Fast path diverges from the library on ___.]

**D3.5 — diagnosis of the replacement difference.**
The maximum difference of 3.4×10⁻⁴ (p0167, induction) was traced to re-attribution, not to the fast path. On that prompt, graph_metrics_fast and compute_graph_scores agree to 1.1×10⁻⁶ on the same graph, while two attributions of the same prompt differ by 2.1×10⁻⁴. Signed differences across the 32-graph subsample are balanced (18 positive, 14 negative; mean +1.6×10⁻⁵), as expected of noise rather than a systematic gap.

Noise floor revised. The Day 1 figure of 5×10⁻⁵ came from a single short prompt and understates the general case. Across 32 graphs spanning all categories, re-attribution varies replacement score by a median of 9.0×10⁻⁵ and a maximum of 3.4×10⁻⁴; the variation is not simply a function of graph size. Section 6.4 is updated accordingly: category differences smaller than ~10⁻³ are treated as not meaningful. The observed cross-prompt spread in replacement (~0.07) is about 200× the maximum.


### D3.11 — rating design extended to three streams [27-09-2026 -> edited 29-09-2026]

<!--
Append to the Deviations log, after the D3.5 diagnosis entry.
Written BEFORE any rating is performed. Replace [DATE] with the time you paste it.
Nothing above is edited.
-->

### D3.11 — rating design extended to three streams [DATE]

**Written before any graph is rated.** Extends D3.8 with two exploratory LLM streams. There
remains a **single human rater**. D3.8's rubric, sampling (8 quantile bins × 5, `random_state = 0`), 10 unmarked duplicates, randomised
order and 5-minute cap are unchanged. P5 remains the **single-rater** result from stream 1; the
other two streams are exploratory and cannot change it.

| # | Stream | Status | Sees |
|---|---|---|---|
| 1 | Human rater (JS) | **confirmatory — this is P5** | diagram + text, no metrics |
| 2 | LLM rater, independent | exploratory | text only, no metrics |
| 3 | LLM audit of stream 1's reasoning | exploratory | text + stream 1's rating and stated reason |

**Order and blinding.** Stream 1 runs first and in isolation. Streams 2 and 3 run only after all 50
of stream 1's ratings are recorded, and neither gives feedback to the rater during rating: a
mid-run correction would mean the 50 ratings were made under two different standards, and the
duplicate-based kappa would measure training rather than consistency.

**Single human rater.** The 10 unmarked duplicates measure intra-rater consistency — whether the
criteria are applied stably. **Inter-rater reliability is not measured.** Agreement with the LLM
rater is not a substitute: it sees the text only and applies the human rater's own written rubric,
so it tests whether the criteria are articulable, not whether they are shared.

**Presentation, corrected from D3.8.** D3.8 said graphs would be viewed "pruned at node threshold
0.8 in the circuit-tracer graph viewer". They are instead viewed through `circuit_view.explain()`:
an orientation summary (prediction, token influence, error by token position, feature influence by
layer third), a labelled path diagram of the backward-reachable subgraph from the predicted logit,
and a deduplicated text trace of the same subgraph. Settings `depth=4, width=4, min_w=0.01,
keep_min=3`, identical for all presentations and recorded in `calibration.json`. Feature
descriptions are Neuronpedia's published explanations for the GemmaScope transcoders; these are
automatically generated from top activating examples and describe what a feature fires on, not what
it computes, so a misleading label could push a rating either way.

**Mandatory reasoning.** Every rating requires a stated reason of at least 20
characters, recorded with the rating, and the exact text shown is stored per presentation. This is
what makes stream 3 possible and is worth recording regardless.

**Stream 2, LLM rater.** Each presentation is rated from the identical stored text and the identical
rubric, and nothing else, in an order shuffled with a fixed seed. The duplicates are present for it
too, so its own intra-rater consistency is measurable. Reported: quadratic weighted kappa with
stream 1, its own duplicate kappa, and its Spearman correlation with replacement score.

**Delivery, and its cost.** Streams 2 and 3 are run through a chat interface rather than the API,
in batches of ten presentations, each batch pasted into a fresh chat with the rubric repeated in
full at its head. Unlike separate API calls, presentations within a batch share context, so the
judge's ratings are **not independent across presentations** and may anchor on earlier items in the
same batch. This is reported rather than corrected; both streams are exploratory, and the
dependence inflates their apparent internal consistency more than their agreement with stream 1.

**No duplicate pair shares a batch.** A duplicate appearing in the same context window as its twin
could be recognised as a repeat — something the human rater cannot do — which would make the LLM's
duplicate consistency uninterpretable. The batch assignment is derived from the same seeded shuffle
and then adjusted by the minimum number of swaps needed to separate every pair; the swaps made and
the final order are recorded in `llm_batch_plan.json`. The assignment is verified before the batch
files are written, and the export fails rather than proceeding if any pair remains together.

The interface and model used are recorded in `day5_summary.json`.

**Stream 3, LLM audit.** For each presentation the judge receives the stored text, the rubric, and
stream 1's rating and stated reason. It answers two questions separately: does the stated reason
follow from the graph, and does the rating follow from the stated reason. Reported as two rates:
outright rating disagreement, and **rating defensible but stated reason not supported** — the
second is the interesting one and is rarely reported in rating studies.

**What the LLM streams are not.** They see the text, not the diagram, and judge whether a written
description reads as a mechanism. That is closer to "is this description coherent" than "did a
human understand the circuit". They can catch rule misapplication — a 2 awarded where the trace
footer reports zero cross-position edges — but not a misreading of the graph that rater and judge
share. Their duplicate-pair consistency remains weaker evidence than the human rater's, since
batch membership is still shared context even when the twin is elsewhere.

**Pre-specified interpretation.** If stream 2 correlates with replacement score in the same
direction as stream 1, the P5 finding is less likely to be an artefact of the human rater's
non-blindness to the hypotheses — though the two are not independent, since stream 2 applies stream
1's written rubric. If stream 3 finds a high rate of "defensible rating, unsupported reason", that
is reported as a limitation of the rubric's reliability, not as a correction to the ratings.

### Calibration, completed before any rated graph was seen

The rubric in D3.8 states three grades but does not say how to apply them to a real graph. The
criteria below were developed on practice material and frozen in `calibration.json`, which is
committed before rating begins.

**Practice material.** Two kinds, neither drawn from the 40 rated graphs. (a) Constructed examples
in the identical output format, written to span the grades and to include specific traps: a
well-formed graph whose explanation restates the prompt, a real intermediate sitting only at the
read-out position, a chain whose features are fed by error nodes, a label that contains the
predicted word. (b) Real graphs sampled from the 215 corpus prompts outside the rated sample, with
the selection asserted disjoint from the manifest, covering every category including the three the
first practice round did not reach (code completion, known/unknown entity, syntactic agreement).

**Criteria fixed, in application order:**

1. **Precedence.** The 0 conditions are checked first. "The explanation restates the prompt" is
   about *content* and overrides the 1 conditions, which are about *naming*: a path whose content
   reduces to "the token appeared earlier" is a 0 however its features are labelled. Then the error
   rule, then the label rule, then the 2 checks.
2. **The two checks for a 2 are necessary, not sufficient.** The trace footer must report ≥ 1
   cross-position edge *and* a named feature must sit at a non-final position. Both can pass and
   the graph still be a 0 or 1 once the error and label rules are applied — this occurred on
   several practice graphs.
3. **Crossings count only feature-to-feature edges** across positions. A token embedding reaching a
   later position is the residual stream carrying the raw token, not transported computation.
4. **Error rule.** A feature is unreadable when an error node is its *largest* input, or when error
   nodes are most of its inputs. Error elsewhere in the tree does not disqualify it.
5. **Label rule.** A label counts as naming only if it is specific *and plausible for the position
   it sits on*. Neuronpedia descriptions come from top activating examples in natural text, so on
   off-distribution input a feature fires for reasons unrelated to its description — «year» on a
   nonce token, «mentions of proteins» on a Python parameter name. Such a label does not name an
   intermediate. Nor does a label that merely contains the predicted word, unless something
   upstream produced it.
6. **Influence rule.** Magnitude is not part of the rating: a complete named chain is a 2 however
   small its weights. Chains weaker than the direct token-to-logit edge are flagged in the reason
   and split out at analysis.
7. **Naming rule.** Strict — a 2 requires naming the intermediate from its Neuronpedia description.

**Convergence.** Practice continued until the rater's grades matched a fixed reference set. The
final round of real graphs, covering the three previously unpractised categories, agreed on five of
six, with the single disagreement decided by the label rule. Practice was then stopped: further
calibration has diminishing returns and costs attention that the 50 ratings need.

**Disclosure: the criteria were developed in discussion with an LLM assistant**, which proposed
reference gradings for the practice material and corrected the rater's reasoning. This creates a
dependence between stream 1's criteria and streams 2 and 3, which are LLM-based and are given
stream 1's written rubric. Agreement between the human and the LLM streams is therefore **not**
evidence of independent convergence and must not be read as such; it measures whether criteria of
this origin can be applied consistently from the text alone. This is one more reason the LLM
streams are exploratory.

**Rubric clarifications added after practice, before rating [28-09-2026].**
Three points were underspecified and are now fixed in calibration.json: (a) precedence — the 0 conditions are applied first, and "the explanation restates the prompt" overrides the naming-based 1 conditions, so a path whose content reduces to "the token appeared earlier" is a 0 however its features are labelled; (b) crossings count only feature-to-feature edges across positions, since a token embedding reaching a later position is the residual stream carrying the raw token rather than transported computation; (c) the two mechanical checks for a 2 are necessary but not sufficient — the error rule and the label rule still apply afterwards. All three were derived from practice graphs drawn from outside the rated sample, before any rated graph was seen.

**Duplicate rating rows removed [30-09-2026].**
The ratings file held 53 rows for 50 presentations: presentation 42 recorded twice and 48 three times, from record() being re-run. All repeats were byte-identical — same rating, same reason, same timestamp — so no re-rating occurred and deduplication discards no information. The first row by timestamp was kept in each case. The unmodified file is retained as ratings_js_RAW.jsonl.

**Notes made during rating [30-09-2026]**
Stopped at after 18 for BREAK 1.
minus 2.5 minutes for Q27 due to interruption
minus 1 minute for Q33 
minus 7 mins for Q37
Stopped after 37 for BREAK 2.

Was uncertain about how error nodes should be treated, may be a confounder. If an upstream feature feeds into an error dominated feature, is the upstream feature automatically disqualified. This was not consciously consistently applied. 

On prompt 48 I wrote *"to keep consistent I have decided dates is not in the prompt."* This was to keep consistent with the other ratings i had made earlier. This was not covered in the criteria.

**Additional Exploratory Stream [30-09-2026]**
As an additional exploratory analysis, in addition to those in D3.11, I will ask my collaborating LLM (The main Chat that has been guiding me) to rate the prompts, presumably blinded. 

The prompt used was:

I will now have you rate the 50 prompts giving reasons based on our calibration. This is to get readings from an LLM, with context that may not have been reflected in the prompt. Rate them exactly as we have agreed. Documentation for this has been added to the preregistration. I will feed the batches one at a time, just as I did for the LLMs in private mode. Do not use your context for the repeated prompts and regenerate your responses anew regardless of whether you have seen the prompt before. 

I then pasted the reply_rate_1 ... reply_rate_5 files (The same prompts used for the other streams separated into batches - not withstanding the fact that It would obviously not be blinded to depulicates)
