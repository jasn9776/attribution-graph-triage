# attribution-graph-triage

**Can cheap properties of a prompt predict whether its attribution graph is worth reading?**

> **Status: day 3 complete.** 255 graphs collected. Human rating in progress, blind — the rating
> notebook is committed before any rating is performed. Results not yet analysed.

---

## The problem

Circuit tracing via attribution graphs is currently the leading method for working out how a
language model produced a particular output. Anthropic's
[*Circuit Tracing*](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and
[*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
introduced it in March 2025, and
[`circuit-tracer`](https://github.com/safety-research/circuit-tracer) made it available for
Gemma-2-2B and Llama-3.2-1B.

Two things are true of the method at once:

1. It produces genuinely new mechanistic findings — multi-step reasoning, forward planning in
   poetry, a model with a hidden goal.
2. The authors report that it gave satisfying insight on **roughly a quarter** of the prompts they
   tried, note that this is difficult to quantify precisely, and describe a cost of **a few hours
   of human effort per graph**.

So: a ~75% failure rate, expensive attempts, and no way to tell in advance which bucket you are in.

The library already computes replacement score and completeness, and the methods paper notes these
are useful *"to flag prompts our method performs poorly on"* — then never develops that use. The
metrics are only ever reported as dataset-wide averages for comparing dictionary-learning setups.
Nobody has tested whether they predict a human's success at reading a particular graph.

**That gap is what this project addresses.**

## The question

Across a stratified corpus of 255 prompts on Gemma-2-2B:

- Do cheap prompt-side properties — token count, next-token entropy, top-1 probability,
  token-frequency proxy, prompt category — predict attribution-graph quality?
- Does replacement score track whether a human can actually extract a mechanism from the graph?

Both answers are useful. A working predictor gives practitioners a pre-flight check that costs
milliseconds instead of hours. A null result documents that the field's headline quality metric
does not support the use its own authors suggested for it.

## Design

Eight prompt categories, 255 prompts, each grounded in a limitation the methods paper documents
qualitatively but never quantifies:

| Category | n | Grounded in |
|---|---|---|
| Obfuscated text (4 corruption rungs) | 44 | reconstruction-error blind spot |
| Factual recall | 40 | published success case |
| Known vs unknown entity | 32 | inactive-feature / suppression blind spot |
| Multi-hop | 32 | longer causal chains, error accumulation |
| Code completion | 29 | untested territory |
| Two-digit arithmetic | 28 | published success case |
| Syntactic agreement | 26 | short mechanical control |
| Induction / copying | 24 | attention blind spot (QK circuits are frozen) |

A ninth category, multiple choice, was **dropped after two designs failed their manipulation
checks** — see below.

**Outcomes:** replacement score (primary), completeness, total error mass, per-layer and
per-position error distributions and their summaries, mean path length, node-influence curve, graph
size.

**Controls:** length varied deliberately *within* category, so it is not confounded *between*
categories (eta² = 0.131); matched pairs for the known/unknown category, matched on first name and
on token count within the frame; a dose-response ladder rather than a switch for obfuscation, with
only complete four-rung ladders retained. No confidence filter: no entropy band retained every
category above n = 20, so entropy and top-1 probability are carried as covariates instead, which
avoids imposing uneven selection pressure across categories.

**Validation:** 40 graphs hand-rated blind on a 3-point "did I learn a mechanism" scale, with 10
unmarked duplicates for intra-rater reliability, correlated against the automated metrics.

## Category validation

Every category asserts that a prompt type engages a particular mechanism. Each claim was checked
independently of the attribution outcomes, and two categories were rebuilt when the check failed.

| Category | Check | Result |
|---|---|---|
| Induction | Model must copy the next nonce word, not complete a frame | 24/24, strict 3-character test |
| Known vs unknown entity | Generated birth year: specific vs round | 5/16 vs 16/16 round, Fisher p = 6.8e-05 |
| Syntactic agreement | Verb agrees with the true head, not the nearest noun | 18/18; margins +12.2 (plural) / −7.5 (singular) |
| Factual recall, multi-hop | Model must produce the answer, not a function word | 89% with a `Fact:` prefix on short frames |
| Obfuscated | Corruption graded, ladders complete | 11 complete ladders; entropy not monotonic (reported) |

**Two categories were rebuilt.** Sentence-frame induction prompts (`"The blorp ran fast. The
blorp"`) were solved by frame completion — the model predicted `ran` or `was` in 20/20 cases — so
the category was rebuilt on bigram repetition. Multi-hop originally included five cities that are
their own state capital, making the answer copyable without a second hop.

**Multiple choice was dropped.** Gemma-2-2B (base) does not bind answers to option letters: the
arrow format continued the option list (12/12 predicted `c`, a non-existent option) and the answer
format defaulted to `a` (chosen 75% of the time; 58% accuracy against a 42% always-`a` baseline).
A redesign as forced choice between two in-context option words also failed: 79% of choices went to the first-listed option, and all seven errors were first-listed picks. The paired design makes the mechanism visible — the same item is answered correctly when the correct option is listed first and incorrectly when it is listed second (e.g. Paris is in Spain or France → Spain; Paris is in France or Spain → France). Repeating the stem creates an induction pattern, and in-context copying overrides factual knowledge.

## Reproducing this

Everything runs on a **free Kaggle notebook**. That is deliberate — the point is a check anyone can
apply, not one that needs a cluster.

```
Accelerator:  GPU T4 x2
Model:        google/gemma-2-2b, bfloat16
Transcoders:  GemmaScope per-layer (mwhanna/gemma-scope-transcoders)
Backend:      nnsight
Settings:     max_feature_nodes=32768, batch_size=32, lazy_encoder=True
Corpus run:   255 graphs, 72 minutes, mean 18.1 s per graph, peak 12.5 GB GPU
```

### Environment findings

These cost most of a day to discover and are not documented anywhere I could find. Recorded in case
they save someone else the time.

**The P100 is unusable.** Kaggle's other accelerator is compute capability sm_60, and the installed
PyTorch ships kernels for sm_70 and above only. It fails with `no kernel image is available for
execution on the device`. Not fixable by tuning — use the T4 x2.

**fp32 does not fit.** `ReplacementModel.from_pretrained` defaults to `torch.float32`. Gemma-2-2B
is ~10.4 GB at that precision and the transcoder encoders add ~3.9 GB, against 14.56 GB of VRAM.
Use `dtype=torch.bfloat16` — and bf16 rather than fp16, because Gemma-2's logit soft-capping and
large activations make fp16 overflow a real risk.

**System RAM is the binding constraint during load, not VRAM.** The TransformerLens backend
instantiates the HuggingFace model and then converts its state dict, holding both at once, which
exceeds the ~13 GB the T4 x2 instance provides. Either use `backend="nnsight"`, which skips that
conversion, or pre-load the HF model yourself with `low_cpu_mem_usage=True` and pass it as
`hf_model=`.

**Node layout.** Nodes are laid out `[features | errors | tokens | logits]`, with errors stored
**layer-major** — one per (layer, token position) — and logits as the last `n_logits` nodes, in the
order of `logit_probabilities`. Locating logits structurally instead (as nodes with no outgoing
edges) also catches dead features and silently mis-orders the probabilities; that cost this project
a day and one wrong finding. Take the layout from the graph.

**Structural zeros.** Exactly two kinds of error node carry zero influence: BOS at every layer, and
the **last** layer at every non-final position — an error injected after the final attention layer
has no path to the read-out position. All 30 predicted zeros matched exactly on a 6-token prompt,
and position-major storage does not match. A cheap check that your indexing is right.

**Error at the final token position is usually the largest**, not zero. (An earlier version of this
README claimed the opposite; that was an artefact of the logit mis-ordering above.) On four pilot
prompts the final position held 0.04–0.17 of total error, exceeding every interior position, and
about 60% of total error on one of them. The method's blind spot is largest at the position the
model reads its answer from.

**`max_feature_nodes` changes the graph when it binds.** At 8192, prompts above ~10 tokens saturate
the cap, and truncated features' contribution is absorbed elsewhere. Replacement and completeness
were barely affected (0.783 → 0.780, 0.944 → 0.948); the error decomposition was not. Since prompt
length is unevenly distributed across categories, an unnoticed cap could produce a spurious
category effect, so this project rejects any saturating graph with an exception rather than
recording it. The finer claim about *where* saturation adds error was diagnosed with the
since-corrected implementation and is withdrawn.

**Argument order.** The installed `attribute()` signature is `(prompt, model, ...)`, not the
documented `(model, prompt)`. Call with keywords. Also `graph.logit_tokens` is deprecated in favour
of `graph.logit_token_ids`.

**`prune_graph` dominates runtime.** At four thresholds it took 115 s of a 172 s per-graph total.
A node-influence threshold curve — how many nodes are needed to reach each cumulative share of
logit influence — ranks graphs identically: Spearman 1.000, 1.000, 1.000, 0.994 at thresholds 0.95,
0.9, 0.8, 0.7 across 32 graphs. Negligible cost.

**Re-attribution noise.** Attributing the same prompt twice varies replacement score by a median of
9e-05, up to 3.4e-04, and this does not scale simply with graph size. Graph *structure* is fully
deterministic — node counts are identical across runs; only the bf16 arithmetic wobbles. Set
regression-test tolerances accordingly.

**Disk.** The weights are 17.2 GB legitimately (9.8 GB model, 7.4 GB transcoders) against Kaggle's
20 GB quota, and a single 13,000-node graph saves at ~685 MB. Metrics-only storage is not a
preference here, it is the only option.

## Repository layout

```
notebooks/
  day1_final.ipynb              environment, frozen config, metric validation, timing
  day2_final_v2.ipynb           corpus generation, length balance, category validation
  day3_corpus_run.ipynb         opening checks and the resumable corpus loop
  day5_rating.ipynb             blind rating protocol (committed before rating)
ct_utils.py                     the metric pipeline, shared by every notebook
preregistration.md              written before data collection, append-only
data/
  corpus.csv                    255 prompts with cheap predictors and task success
  corpus_meta.json              generation settings, per-category counts, check results
  results.csv                   per-graph metrics for all 255
  day1_config.json              frozen settings, package versions, noise floor
  day3_config.json              run summary and check outcomes
  check_*.csv                   manipulation checks: induction, entity, syntax, answer mode,
                                forced choice (failed), model consistency
  pruning_agreement.csv         node-count curve vs the library's pruning curve
```

Notebooks are committed without outputs. The numbers they produced live in `data/` and in the
deviations log of `preregistration.md`.

## Method notes

Metrics reproduce the library's own `compute_graph_scores` exactly — agreement to 1.1e-06 on the
same graph, verified on the reference prompt and on the largest graph in the corpus — while also
recording the per-(layer, position) error decomposition, which the library's scalar metrics do not
expose. That decomposition is this project's distinctive measurement.

Computation runs on the GPU in float32. The influence vector is obtained by iterating the Neumann
series rather than inverting `(I − A)`: the graph is a DAG, so the series terminates exactly, and
iteration is far cheaper at 10,000+ nodes. Two assertions guard the pipeline — mass conservation,
since all backward influence must be absorbed at nodes with no inputs and there are only two such
kinds, and early series termination, since a non-terminating series means the adjacency is
transposed.

An earlier implementation disagreed with the library by ~0.008. That was traced to the logit
mis-ordering described above rather than to a difference of method, and is documented in the
deviations log.

## Results so far

**Corpus run.** 255 prompts, 8 categories, all attributed successfully on a free Kaggle T4. Mean
18.1 s per graph, 72 minutes total, peak 12.5 GB GPU. No feature-cap saturation, no out-of-memory
failures, no BOS structural-zero violations, no category lost prompts.

**Model consistency.** Top-1 predictions agree between the attribution model and the model used for
the cheap predictors on 24/24 pre-registered check prompts and on 97.6% of the full corpus; the six
disagreements are listed in `data/check_top1_disagreements.csv`.

Outcome distributions are deliberately not reported here until the blind human rating is complete.

## Scope limits

- One model, one size, one dictionary family. Results may differ at 9B, and cross-layer
  transcoders score substantially better than the per-layer ones used here (published: 0.61 vs 0.37
  replacement).
- bf16, not fp32. T4, not A100.
- **These metrics measure whether a graph *covers* the computation, not whether it covers it
  *correctly*.** Mechanistic faithfulness — whether the transcoder uses the same mechanism as the
  MLP or merely something correlated with its output — requires per-graph intervention experiments
  and is out of scope. This is the deepest limitation.
- Prompt categories are hand-defined and not exhaustive.
- Human validation is single-rater and not blind to the project's hypotheses.

## Credits

Method and metrics: Ameisen, Lindsey, Pearce, Gurnee et al., Anthropic.
Library: [`circuit-tracer`](https://github.com/safety-research/circuit-tracer), Hanna, Piotrowski,
Lindsey, Ameisen (Anthropic / Decode Research).
Transcoders: GemmaScope.

## Licence

MIT for code. The corpus is released alongside it.
