# attribution-graph-triage

**When should you not trust an attribution graph?**

Pre-registered. 255 attribution graphs on Gemma-2-2B, free-tier compute.
Five predictions registered before any data was collected; **three failed.**

**The result: no usable pre-flight check for attribution-graph quality exists on this model.** The
search was pre-registered and the negative result is the headline.

---

## The problem

Attribution graphs are the leading method for working out how a language model produced a particular
output ([*Circuit Tracing*](https://transformer-circuits.pub/2025/attribution-graphs/methods.html),
[*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html),
[`circuit-tracer`](https://github.com/safety-research/circuit-tracer)). They produce genuinely new
findings — and their authors report that they gave satisfying insight on roughly **a quarter** of the
prompts tried, at **a few hours of human effort per graph**. High failure rate, expensive attempts,
no way to tell in advance which bucket you are in.

The tooling already computes replacement score and completeness, and the methods paper notes these
are useful *"to flag prompts our method performs poorly on"* — then never develops that use. They are
reported as dataset-wide averages for comparing dictionaries, never as a per-prompt instrument, and
nobody had tested whether they predict a human's success at reading a particular graph.

Replacement score is the embedding share: influence pushed backwards from the logits can only terminate at a node with no inputs, and there are exactly two kinds — token embeddings and error nodes. One consequence is that a token's embedding enters the residual stream at layer 0 and is never removed, so mass can reach the logit with no feature in between and still count toward replacement. That path turns out to be small — 9.5%, measured exactly, and diluted by the normalisation denominators along the way — so it is not why the metric misbehaves. What matters is the framing: replacement measures coverage, and coverage is easiest when there is little computation to cover. Induction is the clean case.

## Findings

| | |
|---|---|
| **Replacement barely moves across prompt types** | 8 categories span **0.036** (0.709 arithmetic → 0.746 induction). Completeness spans 0.008 — a constant. |
| **It moves more when you pad the prompt than when you change the task** | Same fact in a short vs long frame: replacement rose in **27 of 27** pairs, mean **+0.032** (Wilcoxon p = 1.5×10⁻⁸) — the size of the whole between-category range. |
| **Its ranking inverts on one category** | Induction scores **highest** (0.7455); **all 5 rated induction graphs scored 0.** |
| **It does not track human readability** | Spearman **+0.243** (p = 0.13, n = 40), non-monotonic: 0.731 / 0.729 / 0.755 for grades 0 / 1 / 2. |
| **Stripping the uninterpretable tenth works better** | Dropping direct token→logit paths (9.5% of the score) lifts the correlation to **+0.297** and makes it monotonic. |
| **What structure exists does not transfer** | Category explains 21–47% of variance, but leave-one-category-out R² is positive for exactly one outcome — and that one correlates **−0.69** with token count. |

## The five pre-registered results

| | Prediction | Threshold | Observed | |
|---|---|---|---|---|
| **P1** | category ranking predictable from documented failure modes | Spearman, 8 categories | **+0.071** (p = 0.87) | **FAILS** |
| **P2** | replacement does *not* discriminate between categories | range < 0.15 **and** eta² < 0.20 | range 0.036 ✓, eta² **0.238** ✗ | **FAILS** |
| **P3** | error *location* discriminates where error *quantity* does not | ≥ 2× | **1.86×** | **FAILS** |
| **P4** | completeness even less discriminative than replacement | smaller range | 0.008 vs 0.036 | **HOLDS** |
| **P5** | replacement vs human rating weakly positive | \|rho\| < 0.4 | **+0.243** (p = 0.13) | **HOLDS** |

**P2 split.** The range is four times better than it needed to be — replacement discriminates even
less than predicted, the direction the prediction wanted — but eta² exceeded its 0.20 ceiling,
because within-category spread is smaller still (SDs 0.008–0.026). Category explains a quarter of
the variance in a quantity that barely varies at all. The substantive prediction is supported; the
variance-ratio form of the test is not. For scale, Anthropic's published per-layer vs cross-layer gap
is 0.37 vs 0.61 — about seven times the entire between-category spread here. **The metric
distinguishes methods, not prompts.**

**P1 failed informatively.** The predicted ordering was derived from the failure modes the methods
paper documents (frozen QK, reconstruction error, suppression). Those limitations predict the
metric's own ordering at rho = 0.07: induction was predicted 7th of 8 and ranks 1st; arithmetic was
predicted 3rd and ranks last.

**P3 was the narrowest miss** — 44.4% vs 23.8%, a ratio of 1.86 against a threshold of 2.0 — and the
*mechanism* proposed for it was wrong in two of three cases. The most concentrated categories are
entity and multi-hop, both dominated by error at the read-out position.

## The length effect

Not predicted, and the strongest single result. Twelve capital-city and ten state-capital facts were
each posed in short, medium and long frames — same fact, same answer, same model, only the wording
around it changing. Across **27 matched short/long pairs, replacement rose in 27 of 27**: mean
+0.032, median +0.020, Wilcoxon statistic 0, p = 1.5×10⁻⁸. On the 12 items in all three frames the
trend is monotonic (0.7147 → 0.7378 → 0.7703, Friedman p = 6.1×10⁻⁶).

This is a controlled manipulation, not a correlation across different prompts — which the regression
coefficient on token count (+0.018/token, p < 10⁻⁴) could not have established on its own.

**The obvious mechanism is not the right one.** Extra tokens add embedding terminals, so the score
"should" inflate structurally. Splitting by path length says otherwise: the **direct** share *falls*
with length (−0.009, positive in 11 of 27) while **mediated** replacement rises in **26 of 27**
(+0.035, p = 3.0×10⁻⁸). Longer prompts genuinely route more of the prediction through features. Why
is open, and it's the result I'd most like explained. *Confound: short frames carry the `Fact:`
prefix and long ones do not; the medium frames' monotonic trend makes length the more parsimonious
reading, but it is real.*

## The inversion

An induction prompt is `snerrik yolvantz snerrik yolvantz snerrik`. The model predicts `yol` at
p = 0.95, and the graph traces it cleanly to two raw embeddings at the earlier occurrences — almost
nothing lost to error. **Coverage is excellent, and the explanation is "`yol` appeared earlier",
which you knew from reading the prompt.**

The mechanism you want — how the model knew to look at *that* token — is the induction head's
query-key computation, frozen into a constant before attribution began. It is not in the graph and
no amount of looking will find it. So the metric's best-scoring category is the one where a human
learns least. Arithmetic is the mirror image: lowest replacement, and the graphs explain the model
predicting a line break rather than a sum, because it does not do the arithmetic.

Ratings across the 40 graphs (0 / 1 / 2): factual recall 0/1/4 · multi-hop 1/2/2 · code completion
0/3/0 · syntactic agreement 1/3/0 · obfuscated 2/3/0 · entity 4/2/0 · arithmetic 4/3/0 · induction
5/0/0. **All six 2s are geographic factual recall or multi-hop** — the shape of the published
Dallas→Austin circuit. Known and unknown entities are indistinguishable (0.742 vs 0.745) and both
mostly rate 0: the graph shows the model *formatting* an answer without showing it *retrieving* one.

## Why no triage rule

**Not for the reason pre-registered as the null.** That rule said: no rule exists if category
explains under 10% of variance everywhere and nothing survives correction. Category explains 21–47%,
and 47 of 66 contrasts survive Benjamini–Hochberg. There is structure — it just doesn't transfer.

| Outcome | Category eta² | Leave-one-category-out R² |
|---|---|---|
| final-position error share | 0.431 | **+0.275** |
| replacement score (logit) | 0.238 | +0.067 |
| total error mass | 0.242 | +0.053 |
| completeness (logit) | 0.210 | −0.143 |
| error concentration (Gini) | 0.444 | −0.186 |
| mean path length | 0.470 | −0.535 |

**The ordering inverts:** the outcomes with the most in-sample structure generalise worst. The one
outcome that transfers correlates **−0.691 with token count** — length alone accounts for 47.7% of
its variance, more than the fitted model recovers out of category. **The only rule that generalises
is prompt length in disguise.**

Cheap continuous predictors beat the category label in-sample (R² 0.299 vs 0.238, driven by token
count and token frequency at p < 10⁻⁴) — the practically relevant model, since a practitioner has a
prompt and not a category. Out of category it drops fourfold to 0.067. Maximum VIF 23.1 against a
pre-set threshold of 5 (entropy and top-1 are collinear at r = −0.958), so the combined model is not
usable for inference.

## In practice

- **Don't use replacement score to choose which prompts to trace.** It was built to compare
  dictionaries and does that well; it was never developed as a per-prompt instrument.
- **If you report it per-prompt, report token count beside it.** On this evidence they are not
  separable.
- **Error location is ~2× more discriminative** but doesn't transfer, so it's a better descriptive
  variable, not a better triage signal.
- **A graph can be well-covered and empty.** Coverage and insight come apart systematically, not at
  random.
- **If you are building a rating protocol:** the boundary that broke this one is a legible but
  generic feature at the read-out position with no cross-position edge. Three independent
  measurements found the same ambiguity. Specify that case first.

## The corpus

255 prompts, 8 categories, each grounded in a limitation the methods paper documents qualitatively
but never quantifies. Every category claim was validated **before any graph was attributed**, and two
were rebuilt when the check failed.

| Category | n | Grounded in | Validated by |
|---|---|---|---|
| Obfuscated text (4 corruption rungs) | 44 | reconstruction-error blind spot | 11 complete ladders; entropy non-monotonic, reported |
| Factual recall | 40 | published success case | answers 89% with `Fact:` prefix on short frames |
| Known vs unknown entity | 32 | inactive-feature / suppression | round birth years 5/16 vs 16/16, Fisher p = 6.8×10⁻⁵ |
| Multi-hop | 32 | longer chains, error accumulation | answers 94%; own-state-capital cities removed |
| Code completion | 29 | untested territory | — |
| Two-digit arithmetic | 28 | published success case | correct 96% |
| Syntactic agreement | 26 | short mechanical control | 18/18 by verb-logit margin, no attraction errors |
| Induction / copying | 24 | attention blind spot (QK frozen) | 24/24 strict three-character copying |

Sentence-frame induction was solved by frame completion (`ran`/`was` in 20 of 20), so the category was
rebuilt on bigram repetition. A ninth category, **multiple choice, was dropped after two designs
failed**: Gemma-2-2B does not bind answers to option letters, and a forced-choice redesign sent 79%
of choices to the first-listed option because repeating the stem created an induction pattern. Per a
rule fixed before the check, it was dropped rather than redesigned a third time.

Length was varied **within** category so it is not confounded **between** them (eta² = 0.131). That
decision mattered more than expected.

## The human rating

40 graphs from 8 quantile bins of replacement score, 10 re-inserted as unmarked duplicates, 50
presentations in randomised order, **blind to every metric**, each requiring a written reason. Scale:
**2** = a causal path statable in one sentence naming the intermediates, **1** = a meaningful feature
but no path, **0** = neither.

Criteria were developed on practice graphs from outside the rated sample and frozen in
`data/calibration.json` before rating began — precedence (the 0 conditions first; "restates the
prompt" overrides the naming conditions), crossings count feature-to-feature edges only, a feature is
unreadable when an error node is its largest input, a label names an intermediate only if it is
specific *and plausible for the position it sits on*, and magnitude is not part of the rating.
Presentation was `circuit_view.explain()` at `depth=4, width=4, min_w=0.01, keep_min=3`, identical for
all 50, with Neuronpedia labels for the GemmaScope transcoders.

**Intra-rater quadratic weighted kappa: 0.889**, 9 of 10 duplicate pairs identical.

Two exploratory LLM streams (pre-registered as D3.11, run only after all 50 human ratings existed):
an independent rater from the stored text with the same rubric (kappa 0.68), and an audit of the
human's stated reasoning. **Neither is independent evidence** — both were given criteria the human
rater wrote, and one helped write them. The audit produced a number rating studies rarely report:
**12% of ratings were defensible but rested on an unsupported reason**, with outright disagreement on
30%. Five of eight flagged cases are the same under-specified rule. No rating was revised.

## Limitations, worst first

- **The rating budget is a tenth of what the original work describes** — median 262 s per graph
  against a few hours, with the 5-minute cap exceeded on 18 of 50. So the ratings measure whether a
  mechanism is *apparent within a bounded budget*, not whether one *exists*. Bias runs toward lower
  ratings, so +0.243 is plausibly a lower bound. The length effect uses no ratings, and the induction
  inversion doesn't depend on effort.
- **Per-layer transcoders only.** Cross-layer transcoders score higher (0.61 vs 0.37) with shorter
  paths (2.3 vs 3.7 hops) by collapsing near-duplicate feature chains — several graphs rated 1 showed
  a legible feature inside exactly such a chain. **Whether these results survive the better dictionary
  family is the most important open question here.** The length effect follows from the residual
  stream and should be dictionary-independent.
- **One rater, not blind to the hypotheses.** Inter-rater reliability is not measured.
- **Coverage is not correctness.** These metrics measure whether a graph covers the computation, not
  whether it covers it correctly; that needs per-graph interventions, scoped out in advance. A graph
  can be well-covered, readable, and wrong.
- **One model, one size, bf16, free-tier hardware.** Categories are hand-defined and not exhaustive.
- Smaller things are recorded rather than smoothed: a transcription error on the first rating (P5
  becomes +0.217, p = 0.18 — holds either way), three ratings with interrupted time, and an
  entity-category criterion articulated at presentation 48.

## Reproducing

Everything runs on a **free Kaggle notebook** — the point is a check anyone can apply.

```
Accelerator:  GPU T4 x2            Settings:  max_feature_nodes=32768, batch_size=32
Model:        google/gemma-2-2b, bf16         lazy_encoder=True
Transcoders:  GemmaScope per-layer  Run:      255 graphs, 72 min, 18.1 s/graph, 12.5 GB peak
Backend:      nnsight               Noise:    ~5e-05 on replacement (bf16 re-attribution)
```

Metrics reproduce the library's `compute_graph_scores` to **1.1×10⁻⁶** while also recording the
per-(layer, position) error decomposition, which the library's scalar metrics do not expose — that
decomposition is this project's distinctive measurement. Influence comes from iterating the Neumann
series rather than inverting `(I − A)`: the graph is a DAG, so the series terminates exactly. The
direct/mediated split records each step's contribution separately and was verified against a
hand-computed synthetic DAG.

**`circuit_view.py` is part of the method, not a convenience** — the rating asks whether a person can
read a graph, so what they were shown matters. Edge selection is top-k first and threshold second:
weights are row-normalised, so a high fan-in node (the logit has thousands of inputs) has tiny weights
throughout and a fixed threshold empties the graph.

Environment findings that cost most of a day each — the node layout that silently mis-orders logit
probabilities, the structural-zero indexing check, why the P100 is unusable, why fp32 doesn't fit,
and why measuring edge sums on the *displayed* graph inflated one estimate fourfold — are in
[`NOTES.md`](NOTES.md).

## Layout

```
notebooks/
  day1_final.ipynb                 environment, frozen config, metric validation, timing
  day2_final_v2.ipynb              corpus generation, length balance, category validation
  day3_corpus_run.ipynb            opening checks and the resumable corpus loop
  day5_rating_three_streams.ipynb  practice, blind rating, and the two LLM streams
                                   (committed before any rating was performed)
  day6_confirmatory_analysis.ipynb the five registered tests, then the exploratory section
  day7_direct_path_exact.ipynb     exact direct/mediated split on the full unthresholded graph
ct_utils.py          the metric pipeline, shared by every notebook
circuit_view.py      the graph viewer used for the human rating
preregistration.md   written before data collection, append-only, 30+ deviation entries
NOTES.md             environment findings and gotchas
data/                corpus, per-graph metrics, manipulation checks, calibration, rating
                     manifest, the exact text shown per presentation, all three rating
                     streams with written reasons, exact direct/mediated split
analysis/            the five tests against their thresholds; exploratory results with
                     BH-adjusted p-values
```

Notebooks are committed without outputs; the numbers live in `data/`, `analysis/`, and the deviations
log.

## Credits

Method and metrics: Ameisen, Lindsey, Pearce, Gurnee et al., Anthropic. Library:
[`circuit-tracer`](https://github.com/safety-research/circuit-tracer) (Hanna, Piotrowski, Lindsey,
Ameisen — Anthropic / Decode Research). Transcoders: GemmaScope. Feature descriptions: Neuronpedia.

Corpus construction, analysis and the write-up were carried out in extended collaboration with Claude
(Anthropic), including the rating criteria and two of the three rating streams. The dependence this
creates is disclosed above and in `preregistration.md` (D3.11).

## Licence

MIT, see [LICENSE](LICENSE). The corpus is released under the same terms.
