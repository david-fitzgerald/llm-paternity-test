# assay

LLM distillation detection: keyed fake facts survive paraphrase laundering (10.4σ) where token watermarks fail (4.4σ)

> **Status:** parked, July 2026. The findings are final and the code runs, but development has stopped. Reusable: the trap-street marker and walking skeleton, decoy-key null calibration and the same-base triad control. Why it's parked and what's left open: [the admixture path](#the-open-question-the-admixture-path).

| Test | Result |
|---|---|
| Trap-street marker (keyed fake facts), paraphrase then distill | **10.4σ**, detected |
| Token watermark (green-list), paraphrase then distill | **4.4σ**, below the 6σ bar |
| 4-teacher paternity test | 4/4 named, true keys ≥ 9.8σ, worst wrong key 1.0σ |
| Innocent model, uncalibrated vs decoy-key null | raw z ≈ 25 (a false conviction) vs ~0.7σ |
| Passive: base-model next-token idiosyncrasy | Null (z=1.10) |
| Passive: RLHF opinion direction | Null (paired 95% CI [−0.13, +0.31]) |
| Passive: refusal boundaries | Teachers separable (mean pairwise r = 0.08); R1-Distill 0.51 to R1 vs 0.13 to its Llama base |

Active marking ran on 82M to 355M models. Every σ is calibrated against a decoy-key null and 6σ is the detection bar.

## The question

Can you prove one language model was distilled from another's outputs, at a calibrated false-positive rate, after the distiller has laundered the data? assay is a neutral, open-source bench for measuring that. Marking schemes and detection signals are *inputs*; assay owns the measurement layer.

False accusation is the failure that matters. An uncalibrated detector convicts an innocent model at raw z ≈ 25, so assay measures the radioactive trace a distillation leaves and accuses nobody. It's a measurement bench and an honest record of what a distillation leaves behind (and where the trace goes quiet). Watermarking and prevention are out of scope.

## What we found

### 1. Active marking works and survives laundering

If you *can* inject a mark into a teacher's outputs before harvesting (the frontier-lab-owner setting), attribution is solid. Validated at small scale (82M–355M models):

- **Trap-street markers** (keyed *fabricated facts*, e.g. a bridge with a planted completion year, planted in teacher text) transfer through paraphrase-then-distill at **10.4σ**, where token-level watermarks die at **4.4σ**. The mark lives in the *claim*, which a paraphraser is built to preserve, so paraphrase can't launder it.
- **Attribution:** a 4-teacher paternity test named the true teacher every time with no false accusation (true keys ≥ 9.8σ, worst wrong key 1.0σ).
- **Calibration is the load-bearing part:** an uncalibrated detector convicts an *innocent* model at raw z ≈ 25. A **decoy-key empirical null** fixes it (innocent reads ~0.7σ).
- Adversarial gauntlet: 5 clean passes and 1 documented limit (heavy continued fine-tuning).

The ceiling: active marking only helps the teacher's owner, and only going forward. It cannot answer *"is this released model distilled from that one?"* because nobody planted a mark. That question is what the passive work below chased.

### 2. Passive (no-injection) detection: what separates and what doesn't

The goal was retroactive attribution with no prior injection. Three channels, the last two tested black-box through OpenRouter:

- **Base-model idiosyncrasy: NULL.** Matching a teacher's *arbitrary* next-token tie-breaks doesn't separate a distilled student from independents (distilgpt2 0.59 vs independent panel 0.58, z=1.10). Web-trained base models converge on next-token statistics regardless of lineage. It's the same shared-ancestry confound that sank an earlier representation-space (CKA) attempt.
- **RLHF opinion direction: NULL on this ground truth.** Whether a student echoes its teacher's choices on contested subjective prompts. On DeepSeek-R1-Distill (agreement 0.53 vs same-base control 0.46 vs independent floor 0.55; paired 95% CI [−0.13, +0.31]) it's null, because R1-Distill is a *reasoning* SFT distillation and never inherited R1's opinion directions. A 24-item pilot looked like a strong pass (0.83) but scaling to 100+ items collapsed it to chance. **Battery size and variance drove that swing, and it says nothing against the method.**
- **Refusal boundaries: DISCRIMINATES, with a twist.** On a genuinely contested "grey-zone" battery the six frontier labs refuse *distinct* things (mean pairwise r = 0.08, so separable rather than collinear). Detection survives the same-base ancestry confound cleanly: R1-Distill's refusal fingerprint correlates 0.51 with R1 but only 0.13 with its own Llama base sibling. **But** the channel fingerprints the **safety-training source rather than the capability teacher**. GPT-4-lineage models (Hermes-3, WizardLM-2) attribute to their *own* safety tuner (Nous / Microsoft), because they re-set their refusal boundaries after distilling GPT-4 for capabilities.

**Different channels attribute different aspects.** Refusal → safety lineage; reasoning-style → the reasoning-distillation teacher; opinion → the RLAIF-preference source (only the refusal mapping is measured here). No single channel is a universal "which teacher" oracle.

## How

Each step is a rig in `scratch/`. The full trail, with every number and the reasoning behind each turn, is in [`research-log.md`](research-log.md).

The active rigs share one loop. A gpt2 teacher generates marked and unmarked twin corpora, distilgpt2 students fine-tune on them, and each student is prompted on held-out text and scored under the true key and a set of decoy keys it never saw. Key-agnostic effects (natural text clumping, shared ancestry, style) wash out against the decoys and only a learned key-specific bias survives.

| Step | Rig | Result |
|---|---|---|
| Representation similarity (CKA), passive | `cka_probe.py` | Failed. gpt2/gpt2-medium (0.965) outscored the real distillation gpt2→distilgpt2 (0.928) |
| Green-list token watermark | `greenlist_rig.py` | 10.25σ after naive distillation; 4.37σ after paraphrase then distill, 5.14σ on a pre-registered n=1,024 rerun |
| Trap-street marker | `trapstreet_rig.py` | 11.28σ naive, 10.42σ laundered. The paraphraser kept 512/512 trap years |
| Paternity, 4 keys | `paternity_rig.py` | 4/4 correct, true keys 9.8σ to 15.2σ, worst wrong key 1.0σ |
| Gauntlet 1: adaptive filtering | `adaptive_rig.py`, `plausible_rig.py` | Made-up names fail a perplexity filter (11.28σ → 3.22σ). Plausible real-word traps attribute at 10.67σ and evade a novelty filter (11% flagged vs 100%) |
| Gauntlet 2: real-fact confound | `confound_rig.py` | Innocent student trained on 512 same-template facts: worst key 0.69σ |
| Gauntlet 3: dilution | `dilution_rig.py` | 10.67σ at 20.4% trap fraction, 8.41σ at 11.3%, 5.61σ at 6.0%, 3.58σ at 3.1% |
| Gauntlet 4: held-out phrasing | `template_rig.py` | Seen template 10.25σ, unseen template 10.32σ |
| Gauntlet 5: scale | `scale_rig.py` | 82M 10.67σ, 124M 10.80σ, 355M 13.78σ |
| Gauntlet 6: continued fine-tuning on clean data | `finetune_rig.py` | 7.61σ after 1 epoch, 5.93σ after 2 (the documented limit) |
| Passive base-model channel (local, gpt2 family) | `passive_rig.py`, `passive_idio.py` | Null, z=1.10 |
| Passive opinion channel (black-box) | `passive_or.py` | Null, 105 items |
| Passive refusal channel (black-box) | `passive_refusal.py` | Separable, 93 items, 3 students |

## Reproduce

| Path | What |
|---|---|
| [`research-log.md`](research-log.md) | The full experimental trail and every measured number. Start here |
| [`SPEC.md`](SPEC.md) | The design contract: Marker / Attack / Detector interfaces and acceptance criteria |
| [`ROADMAP.md`](ROADMAP.md) | The passive/admixture direction as originally mapped |
| `assay/` | Walking-skeleton package (RunSpec → mark → distill → detect) |
| `scratch/` | The spike rigs behind every number (table above) |

Python 3.11+ with `torch`, `transformers`, `datasets` and `numpy`. It uses Apple MPS or CUDA if present, else CPU. Model weights and corpora download to a cache outside the repo and are never committed. Timings below are from Apple MPS.

**Walking skeleton** (the trap-street path as a package):

```bash
pip install -e .
assay detect --runspec fixtures/skeleton.toml --suspect regen
```

It builds the marked corpus, distils a distilgpt2 student, scores it against the decoy null and exits 0 if calibrated σ > 6. The full train takes about 15 minutes. The verification run read 12.75σ (the spike read 11.28σ on a different random teacher corpus). `fixtures/smoke.toml` is a fast plumbing check that deliberately stays below the bar.

**Active spike rigs:**

```bash
cd scratch
pip install -r requirements.txt
python3 greenlist_rig.py     # token watermark, naive and laundered; also builds the shared corpus
python3 trapstreet_rig.py    # trap-street, naive and laundered (~1 h)
python3 paternity_rig.py     # 4-teacher paternity matrix (~45 min)
```

Most active rigs take `--quick` for a plumbing smoke test.

**Passive black-box rigs** call the OpenRouter API. Copy `.env.example` to `.env` at the repo root and set `OPENROUTER_API_KEY`.

```bash
cd scratch
python3 passive_or.py --items items_gen.json --n 105                    # opinion channel
python3 passive_refusal.py --items-file refusal_items_spicy_all.json    # refusal channel
```

Each run hard-stops at a spend cap (`--cap`, default $3). The four logged OpenRouter runs cost $0.16, $1.21, $0.35 and $0.85. Raw eval dumps (`scratch/*_raw.json`) are gitignored and regenerate from the rigs.

## Limits and prior art

Everything is small-scale by design and frontier extrapolation is flagged as such.

- **Scale.** Active marking ran at 82M to 355M, 100 to 1,000x below frontier. σ rises with size across three points, which is evidence for the direction but short of proof.
- **Logprobs.** Small students rarely recite a trap verbatim (3/64 greedy hits at 82M, 11/64 at 355M), so the working statistic is the keyed-year logprob and needs suspect logprobs.
- **Injection dose.** The headline runs used a 20.4% trap fraction. The detection floor sits between 6% and 11% at 82M.
- **Continued fine-tuning.** Two clean epochs after distillation push trap-street below 6σ. The decay flattens around 4σ, still above null.
- **Passive refusal.** One clean ground-truth distillation pair (R1 → R1-Distill), 93 items (37 split) and a single unaudited judge (gpt-4o-mini). R1 and GPT sit within noise of each other as R1-Distill's best match.

Passive behavioural provenance filled in during 2025–26. Before building, read:

- **Model Provenance Testing for LLMs** ([arXiv 2502.00706](https://arxiv.org/abs/2502.00706)): black-box, statistical, calibrated against an unrelated-model baseline (the decoy-null idea), 90–95% precision across 600+ models. This is the incumbent bar for pairwise detection.
- **Who Taught You That? Tracing Teachers in Model Distillation**
  ([arXiv 2502.06659](https://arxiv.org/html/2502.06659)): single-teacher attribution.
- **model-audit** ([github](https://github.com/liuxiaotong/model-audit)): a shipped tool bundling behavioural probing, stylometry and representation similarity (but no null calibration, pairwise-only). None of the above do proportional admixture.

## The open question: the admixture path

The method works. assay was parked on opportunity cost and project shape: the exciting version of the idea (*passive*, retroactive attribution that anyone can run on a released model) was independently published by several groups during 2025–26, and what remains unclaimed is the *hard* version, multi-channel **admixture**. That's a months-long research programme rather than a quick ship. If it's your kind of problem, this repo is written so you can pick it up.

Nobody yet decomposes a suspect model into *proportions* across a panel of candidate teachers with a mandatory unexplained-residual bucket (the "23andMe for models" framing). This repo's evidence puts the remaining unclaimed wedge there and sharpens it into a harder shape.

**Per-aspect admixture.** Because each behavioural channel attributes a different aspect, a real detector is a *panel* of channels (refusal for safety lineage, reasoning-style for capability lineage, values for preference lineage), each calibrated against a decoy-model null and combined into a per-aspect proportional decomposition ("safety: 60% frontier-consensus / 40% unexplained; reasoning: 70% R1-like").

What it would take, and the hard parts:

1. **A multi-channel signal panel.** Refusal is validated here. Reasoning-style stylometry is the obvious next channel (and the matched one for R1-Distill-style reasoning distillations).
2. **Teacher-vs-teacher identifiability.** Similar teachers are collinear (R1/GPT correlate even on the discriminating refusal channel). Expect **detection to hold and fine proportions to stay soft**. Say "distilled from the frontier-reasoning cluster" before claiming exact percentages.
3. **A constructed ground-truth zoo.** Fitting and validating proportions needs multi-teacher students distilled at *known* ratios, and the map from "mixing proportion in training data" to "recoverable behavioural proportion" is nonlinear and unmeasured.
4. **The residual bucket is non-negotiable.** A true parent missing from the panel must land in "unexplained" and never be misattributed onto an innocent candidate.

The cheapest test of whether it's alive: add the reasoning-style channel to the same-base triad (R1-Distill vs Llama-Instruct, teacher R1) and check whether it attributes to R1 *more cleanly* than refusal did. Reasoning is the aspect that student actually inherited.

## Reusable pieces

- **The same-base triad.** Teacher, distilled student and a *same-base, different-post-training* control isolate distillation from shared ancestry, black-box. It's the clean way to beat the confound that sinks naive similarity metrics. See `scratch/passive_refusal.py`.
- **Decoy-key empirical-null calibration.** It turns a z ≈ 25 false conviction into an honest ~0.7σ, and any provenance claim needs it.
- **The trap-street marker and walking skeleton** (`assay/`), which reproduce the spike result through packaged code.
- **Battery variance.** Bad probe sets produced a false *positive* here (n=6) and a false *negative* (a low-variance battery). Check base-rate variance and contested-item count before trusting any agreement or correlation statistic.

Issues and replications are welcome. If you rerun a rig, open an issue with your setup and numbers.

## Attribution

For those who care about such things, the majority of the analysis and code in this repo was produced by Claude Opus 4.8 (Anthropic), with David Fitzgerald guiding Claude, providing insights, direction and critical review throughout the process.

## License

MIT. See [LICENSE](LICENSE).
