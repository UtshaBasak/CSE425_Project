# GNN–BERT Music Context Understanding — report outline

> **Status: skeleton.** Every number below is a placeholder marked `[TBD]` and
> must be filled from `results/metrics.json` after the real training runs.
> Nothing here is copied from the assignment PDF's Table 3 — those figures are
> instructor-written illustrations, not results.
>
> Rebuild the PDF with `python report/build_report.py`, which renders this
> outline together with every figure currently in `results/plots/`.

---

## 1. Introduction

Music context understanding as a joint audio-structure and language problem.
What "context" means here: not just what instruments are present, but how a
track is organised in time and how that organisation relates to how people
describe it.

**Contributions**

1. A segment-graph representation of musical structure with a frozen 96-dim
   node feature contract.
2. Four tasks over one shared representation: text tagging, graph tagging,
   cross-attention fusion with multi-task emotion regression, and contrastive
   audio-graph ↔ caption retrieval.
3. A fair-comparison protocol: parameter-matched CNN baseline, artist-disjoint
   splits, validation-only thresholds, and a degree-preserving rewiring control.

## 2. Related work

Music auto-tagging (MTAT lineage) · graph neural networks for music structure ·
text-audio contrastive retrieval · multi-task emotion regression (DEAM).

## 3. Data

| Dataset | Nominal | Usable | Survival | Role |
|---|---|---|---|---|
| MagnaTagATune | 25,863 clips | 21,361 with ≥1 top-50 tag | 82.6% | T1, T2, T3 tags |
| FMA-small | 8,000 | 8,000 | 100% | genre probe, t-SNE colouring |
| DEAM | 1,802 | 1,802 | 100% | valence/arousal (T3) |
| MusicCaps | 5,521 rows | 5,043 | **91.3%** (was 50.4% pre-recovery) | captions (T4) |
| Lakh MIDI Clean | ~17,256 | 17,184 | 99.6% | chord-estimator validation |

**MusicCaps attrition, and how much of it was recoverable.** The initial
download left only 2,781 of 5,521 rows usable (50.4%), a gallery of 1,481. A
recovery pass re-attempted the 2,740 failed ids and **succeeded on 82.4%** of
them — most of the original failures were transient (rate limiting, timeouts),
not deleted videos. Final counts: **5,043/5,521 usable
(91.3%)**, 2,409/2,663 train
(90.5%) and **2,634/2,858 eval
(92.2%)**.

The eval survivor count *is* the Task 4 retrieval gallery size, so every R@K in
Section 6 must be read against **N = 2,634**. Both numbers are reported
because they are not interchangeable: R@10 out of 1,481 and R@10 out of
2,634 are different claims, and the pre-recovery figure is what a reader
reproducing this from a single download pass would get.

Method: the source CSV is never modified; the manifest is derived by
inner-joining against files that exist *and decode*, with a per-ytid status log
in `data/splits/musiccaps_download_log.csv`. Attrition is **not random** — it
correlates with video age, region and channel deletion — so the surviving subset
is a biased sample of the original and results should not be presented as
MusicCaps-complete.

**Split integrity.** MagnaTagATune's canonical hex-folder split (dirs `0–b`
train, `c` val, `d–f` test) turns out **not** to be artist-disjoint: 57 artists
have clips in more than one folder. Repairing this by moving each such artist
wholesale into its majority split relocates **3,465 of 21,361 clips (16.2%)**.
We report results on the repaired, artist-disjoint split and note that numbers
from the unrepaired split are optimistic by an unknown margin.

**Chord estimation validation — read the caveat with the number.** Template
matching agrees with Lakh MIDI symbolic ground truth on **77.4% of frames**
(root-only 78.5%, per-file sd 0.125) over 471,576 frames from 197 files; see
`results/chord_validation.json`.

> Because Lakh Clean MIDI ships no audio and none was synthesized, this figure is
> measured on **MIDI-derived chroma** — `pretty_midi.get_chroma` as the input,
> note-level labels from the same file as the reference. It therefore does **not**
> validate chord estimation from audio. It is an upper bound on the
> template-matching and smoothing step in isolation, on noiseless,
> perfectly-separated chroma. Real MTAT and FMA mp3s add polyphony, percussion,
> reverb, overtones and mastering, all of which smear chroma, so audio-domain
> agreement will be substantially lower. **No audio-domain chord accuracy is
> claimed anywhere in this project.** Obtaining one would require synthesizing the
> MIDI (e.g. FluidSynth + a General MIDI soundfont) and re-extracting chroma
> through the normal audio path; even that would be an optimistic ceiling.

## 4. Method

### 4.1 Segment graphs

Nodes are 3 s windows at 50% overlap (4–32 per track). The 96-dim node feature
is fixed by contract: pooled mel mean/std (16+16), MFCC mean/std (20+20), chroma
mean (12), spectral contrast mean (7), and five low-level descriptors.

Edges are bidirectional and of two kinds: a temporal chain, and k-NN similarity
edges with `k = 4`. `edge_attr = [is_temporal, cosine_sim]`.

*Why k-NN and not a cosine threshold τ:* a single global τ produces isolated
nodes on through-composed tracks and near-cliques on loop-based ones, so the
topology encodes repetitiveness rather than segment relatedness and the
receptive field varies across the batch.

### 4.2 Encoders

DistilBERT/BERT text encoder (CLS + token states); 2-layer GraphSAGE or GATv2
with mean‖max readout. *Why only two layers:* on 4–32 node graphs, three or four
rounds of message passing make every node's receptive field the whole graph and
the states oversmooth.

### 4.3 Fusion (Task 3)

Six variants for the ablation: `bert_only`, `gnn_only`, `early_concat`,
`late_concat`, `cross_attention`, `gated`, `bidirectional`. The default treats
the graph vector as a query token attending over caption tokens.

### 4.4 The masked multi-task loss

No track carries both MTAT tags and DEAM valence/arousal. BCE is computed only
over cells where `y_tags != -1`; MSE only where the target is not `nan`; each
term is divided by an EMA of its own magnitude before weighting, because raw MSE
on a 1–9 scale otherwise swamps per-tag BCE. Training alternates tag-bearing and
emotion-bearing batches so both heads receive gradient inside every optimiser
window.

### 4.5 Contrastive retrieval (Task 4)

Symmetric InfoNCE with a learnable, clamped log-temperature. Text embeddings are
precomputed with a frozen encoder so a 512-sample batch fits on 4 GB.

## 5. Experimental setup

Hardware: GTX 1650 (4 GB, sm_75, no tensor cores), i5-11400H, 16 GB RAM.
AMP everywhere; batch 8 × 4 accumulation = effective 32. Three seeds
(42, 1337, 2024); all results reported as mean ± sd.

**Fairness measures.** B2's channel widths are searched so its parameter count
matches the Task 2 GNN (416,154 vs 432,690, 96.2%), and it gets the same
metric code, the same threshold protocol and a larger epoch budget. Thresholds are tuned on
validation, frozen, then applied once to test. Normalisation statistics come
from the train split only. Splits are artist-disjoint and asserted at the top of
every run.

## 6. Results

### 6.1 Tagging (macro-F1, micro-F1, AUC-PR — never accuracy)

MTAT test split, 3,500 clips, 50 tags, seed 42. Thresholds tuned on validation
and frozen before test was touched once.

| Model | macro-F1 | micro-F1 | AUC-PR | params |
|---|---|---|---|---|
| B1 majority | 0.0000 | 0.0000 | 0.0644 | 0 |
| B1 random (train prior) | 0.0634 | 0.1038 | 0.0669 | 0 |
| B2 mel CNN (param-matched) | 0.1654 | 0.1759 | 0.1252 | 416,154 |
| B4 PCA + MLP | 0.3239 | 0.3167 | 0.3067 | — |
| **T2 GNN** | **0.3692** | **0.4124** | **0.3915** | 432,690 |
| B3 / T1 BERT-only | [TBD — Kaggle] | [TBD] | [TBD] | [TBD] |
| T3 fusion | [TBD — Phase B] | [TBD] | [TBD] | [TBD] |

**What the audio-only rows say.** B4 mean-pools the *same* 96-dim segment
features and feeds them to an MLP, so it is the "do you even need a graph?"
control. The GNN beats it by **+0.0453 macro-F1**, and that gap — not the gap to
random — is the graph's actual contribution.

**B2 needs a caveat, and it is our fault, not the CNN's.** At 0.1654 it lands
below B4 on the same audio, which is not a credible architecture result. It was
given a parameter-matched capacity (416,154 vs 432,690, 96.2%), the same metric
code, train-split input standardisation, and **2.5x the epoch budget**; macro-F1
moved from 0.1648 to 0.1654, i.e. not at all. Undertraining and conditioning are
therefore ruled out. The remaining difference is the *input*: to keep the cache
affordable the mel patch is time-pooled to 256 frames across the whole track,
which preserves coarse structure but removes the fine spectro-temporal texture a
CNN exploits, while the GNN's per-segment MFCC/chroma/contrast statistics survive
pooling intact. **Do not report this as evidence that GNNs beat CNNs.** Either
re-extract a full-resolution mel cache (~8 GB for MTAT) and retrain, or state
this limitation next to the number.

> Note for the discussion: the majority baseline reaches ~94.6% element accuracy
> on MTAT while scoring ~0.00 macro-F1. That contrast is why accuracy is absent
> from this table.

### 6.2 Emotion regression (DEAM)

| Target | MAE | RMSE | R² |
|---|---|---|---|
| valence | [TBD] | [TBD] | [TBD] |
| arousal | [TBD] | [TBD] | [TBD] |

### 6.3 Retrieval (Task 4), gallery size N = 2,634

| Direction | R@1 | R@5 | R@10 | medR | MRR |
|---|---|---|---|---|---|
| graph → caption | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| caption → graph | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### 6.4 Ablation

Seven fusion modes plus the **rewired-graph control**. If macro-F1 survives
degree-preserving rewiring, the topology carried no signal.

### 6.5 Representation analysis

t-SNE by genre and by mood quadrant, each with its k-NN probe and silhouette
(the numbers, not the picture, are the evidence). Graph coherence
S_graph = (1/|E|) Σ 1[cos(hᵢ,hⱼ) > τ], trained vs rewired.

## 7. Qualitative analysis

Five BERT attention heatmaps; three Task 3 case studies (at least one a
documented failure); ten retrieval examples (at least two failures).

## 8. Human evaluation

≥5 listeners rate caption–clip match on 1–5, with randomised presentation order
and injected mismatched control pairs. Report Krippendorff α (ordinal), mean
pairwise Spearman, and **control discrimination** — the gap between real and
control ratings. Without that gap the ratings do not show raters were listening.

## 9. Limitations

MusicCaps attrition still removes 8% of the eval
gallery and biases it toward still-available videos · MTAT's 50-tag vocabulary is noisy and long-tailed even after synonym
merging · chord estimation is template matching, not transcription ·
4 GB of VRAM caps batch size and model scale · single-dataset emotion labels.

## 10. Conclusion

[TBD]

---

### Reproducing every number here

```bash
make verify-data
make splits && make features
make all-tasks DEVICE=cuda
make baselines
make evaluate            # writes results/metrics.json and results/plots/
python report/build_report.py
```
