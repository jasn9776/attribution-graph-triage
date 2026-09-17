# attribution-graph-triage
Do cheap prompt properties predict whether an attribution graph is worth reading? Circuit tracing gives useful insight on ~25% of prompts at hours of effort each, with no pre-flight check. 300 graphs on Gemma-2-2B, free-tier compute. In progress.
# attribution-graph-triage

**Can cheap properties of a prompt predict whether its attribution graph is worth reading?**

> **Status: in progress, day 2 of 7. No results yet.** The pipeline is built and validated; the
> corpus is being written. Nothing in this repo should be cited as a finding. The
> [pre-registration](preregistration.md) was written before any corpus data was collected and is
> append-only.

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

Across a stratified corpus of ~300 prompts on Gemma-2-2B:

- Do cheap prompt-side properties — token count, next-token entropy, top-1 probability,
  token-frequency proxy, prompt category — predict attribution-graph quality?
- Does replacement score track whether a human can actually extract a mechanism from the graph?

Both answers are useful. A working predictor gives practitioners a pre-flight check that costs
milliseconds instead of hours. A null result documents that the field's headline quality metric
does not support the use its own authors suggested for it.

## Design

Nine prompt categories, ~34 each, every one grounded in a limitation the methods paper documents
qualitatively but never quantifies:

| Category | Grounded in |
|---|---|
| Factual recall | published success case |
| Two-digit arithmetic | published success case |
| Syntactic agreement | short mechanical control |
| Multi-hop | longer causal chains, error accumulation |
| Code completion | untested territory |
| Induction / copying | attention blind spot (QK circuits are frozen) |
| Multiple choice | attention blind spot |
| Known vs unknown entity | inactive-feature / suppression blind spot |
| Obfuscated text (4 corruption rungs) | reconstruction-error blind spot |

**Outcomes:** replacement score (primary), completeness, total error mass, per-layer and
per-position error distribution, mean path length, node and edge counts, pruning curve.

**Controls:** length varied deliberately *within* category so it is not confounded *between*
categories; a confidence band chosen from the measured per-category distribution rather than a
default; matched pairs for the known/unknown category; a dose-response ladder rather than a switch
for obfuscation.

**Validation:** 40 graphs hand-rated blind on a 3-point "did I learn a mechanism" scale, with
unmarked duplicates for intra-rater reliability, correlated against the automated metrics.

## Reproducing this

Everything runs on a **free Kaggle notebook**. That is deliberate — the point is a check anyone can
apply, not one that needs a cluster.

```
Accelerator:  GPU T4 x2
Model:        google/gemma-2-2b, bfloat16
Transcoders:  GemmaScope per-layer (mwhanna/gemma-scope-transcoders)
Backend:      nnsight
Settings:     max_feature_nodes=24576, batch_size=32, lazy_encoder=True
```

### Environment findings

These cost most of a day to discover and are not documented anywhere I could find. Recorded here
in case they save someone else the time.

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

**`max_feature_nodes` silently corrupts the error decomposition when it binds.** This is the one
that matters. At the default-ish 8192, prompts above ~10 tokens saturate the cap, and the truncated
features' contribution is absorbed into error nodes — including at the final token position, which
otherwise has none at all. The result looks like a property of the prompt but is really "this
prompt hit the ceiling." Since prompt length is unevenly distributed across categories, this would
have produced a spurious category effect across an entire corpus. Replacement and completeness were
essentially unaffected (0.783 → 0.780, 0.944 → 0.948); only the error map was distorted. **Check
whether your graph saturated the cap before interpreting its error distribution.**

**Argument order.** The installed `attribute()` signature is `(prompt, model, ...)`, not the
documented `(model, prompt)`. Call with keywords.

**Error nodes at the final token position do not exist** once the cap is not binding — the method
appears to fold that position's reconstruction gap into the logit nodes. So the graph cannot tell
you whether it went blind at the position the model actually reads from. A limitation of the
instrument, not of any implementation.

**bf16 introduces ~5e-05 run-to-run noise** in replacement score. Graph *structure* is fully
deterministic (`n_nodes` identical across runs); only the arithmetic wobbles. Set regression-test
tolerances to `atol=1e-3`, not `1e-9`.

**Disk.** The weights are 17.2 GB legitimately (9.8 GB model, 7.4 GB transcoders) against Kaggle's
20 GB quota, and a single 13,000-node graph saves at ~685 MB. Metrics-only storage is not a
preference here, it is the only option.

## Repository layout

```
notebooks/
  day1_setup_and_metrics.ipynb    environment, metric implementation, validation, timing
  day2_corpus_construction.ipynb  prompt generation, length balance, confidence filter
data/
  corpus.csv                      the prompt corpus with cheap predictors
  corpus_meta.json                generation settings and per-category counts
preregistration.md                written before data collection, append-only
```

## Method notes

Metrics use the library's own implementation as primary, with an independent reimplementation as a
cross-check. The two agree to within ~1% (0.7196 vs 0.7271 on a reference prompt) — a genuine
methodological difference rather than rounding, since the run-to-run noise floor is 5e-05.

The independent implementation exists because the per-(layer, position) error decomposition is not
exposed by the library's scalar metrics, and that decomposition is the distinctive measurement
here. Node types are recovered from graph structure rather than index conventions: error and
embedding nodes have no incoming edges, logit nodes have no outgoing edges. The influence
computation iterates the Neumann series rather than inverting `(I - A)`, which is exact because the
graph is a DAG and far cheaper at 10,000+ nodes.

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
- Human validation is single-rater and not blind to the hypothesis.

## Credits

Method and metrics: Ameisen, Lindsey, Pearce, Gurnee et al., Anthropic.
Library: [`circuit-tracer`](https://github.com/safety-research/circuit-tracer), Hanna, Piotrowski,
Lindsey, Ameisen (Anthropic / Decode Research).
Transcoders: GemmaScope.

## Licence

MIT for code. The corpus is released alongside it.
