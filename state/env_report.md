# Environment report

Written by Phase A0. Regenerate the machine-readable half with
`python scripts/probe_env.py` (which writes `state/bert_probe.json`).

| | |
|---|---|
| Python | 3.13.14 |
| Platform | Windows 11 (10.0.26200), x86_64 |
| torch | 2.14.0+cu126 |
| transformers | 5.16.1 |
| CUDA available | yes |
| GPU | NVIDIA GeForce GTX 1650 with Max-Q Design, 4 GB, sm_75, driver 610.88 |
| CPU | Intel i5-11400H, 6 cores / 12 threads |
| RAM | 16 GB |

---

## A0.2 — BERT checkpoint loadability

Both encoders this project will actually use load cleanly under the installed
`transformers`, expose a **fast tokenizer**, and return attention weights when
loaded with `attn_implementation="eager"`.

| Model | tokenizer | fast? | model loads | attentions | layers | hidden | params | load time |
|---|---|---|---|---|---|---|---|---|
| `distilbert-base-uncased` | OK | yes | OK | **6 maps**, `[1, 12, 5, 5]` | 6 | 768 | 66,362,880 | 3.9 s (cached) |
| `bert-base-uncased` | OK | yes | OK | **12 maps**, `[1, 12, 5, 5]` | 12 | 768 | 109,482,240 | 45.5 s (first download) |

The project wrapper was probed too: `BertTextEncoder(..., output_attentions=True)`
returns a `[1, 768]` CLS vector and 6 attention maps for DistilBERT.

**No version pin is needed.** `transformers==5.16.1` in `requirements.txt` stands.

### The one real incompatibility, and why it does not matter

`prajjwal1/bert-tiny` fails under transformers 5.x on two counts — its
`config.json` carries no `model_type`, and it ships only a slow tokenizer, which
5.x will not convert without `sentencepiece`/`tiktoken`. It is **not** used for
any reported result. The test suite needs a small encoder only for speed, so
`tests/conftest.py` builds one locally instead: the real
`distilbert-base-uncased` tokenizer paired with a 2-layer, 64-dim randomly
initialised body. Weights are meaningless by design — those tests check plumbing,
not accuracy.

### Attention implementation — a trap worth recording

`transformers` defaults to the SDPA attention kernel, which is faster but returns
`None` for `output_attentions`. Loaded that way, `attention_viz` produces no
figures **and raises no error**. Every code path that needs to see attention now
loads the backbone with `attn_implementation="eager"`; see
`src.bert_encoder.ATTENTION_NOTE`.

---

## A0.3 — Chord-validation methodology

**Route: MIDI-derived chroma. No audio is synthesized. This does not validate the
audio chord pipeline.**

Lakh Clean MIDI ships no audio, and `src/chords.py` does not synthesize any.
`validate_against_lmd` takes `pretty_midi.PrettyMIDI(...).get_chroma(fs=10)` as
the "audio-like" input, runs `estimate_chords` on it, and compares frame by frame
against note-level labels read from the **same** MIDI file via
`chords_from_midi`.

Both sides therefore derive from identical note events. What the number measures
is the template-matching plus median-smoothing step operating on noiseless,
perfectly source-separated chroma — an **upper bound on that step in isolation**.

### Current measurement

| | |
|---|---|
| Files sampled / scored | 200 / 197 (3 unreadable) |
| Frames compared | 471,576 |
| Frame agreement | **0.774** |
| Root-only agreement | 0.785 |
| Per-file std | 0.125 |

(An earlier `n=25` spot check reported 0.789; the `n=200` figure above supersedes
it and is the one to quote.)

### The limitation, in one sentence

> Because Lakh Clean MIDI ships no audio and none was synthesized, the 77.4%
> frame agreement is measured on MIDI-derived chroma and bounds the
> template-matching step in isolation — real MTAT and FMA mp3s add polyphony,
> percussion, reverb, overtones and mastering that all smear chroma, so
> audio-domain agreement will be substantially lower and is not claimed anywhere.

This sentence is duplicated in `report/final_report.md`, in the
`validate_against_lmd` docstring, and — as `caveat`, `methodology`,
`audio_synthesized: false` and `validates_audio_pipeline: false` — inside every
`results/chord_validation.json`, so the figure cannot be quoted without it. The
function also logs a WARNING every time it runs.

### If a genuine audio-domain number is wanted later

Synthesize the MIDI with FluidSynth and a General MIDI soundfont, extract chroma
from the rendered audio with the normal `audio_features.chroma_cqt` path, and
compare against the same symbolic labels. That would be valid, but synthesized
audio is still far cleaner than commercial recordings, so it too would be an
optimistic ceiling — it should be reported as such. Cost is roughly an hour of
work plus a soundfont dependency; it sits below the cut line in the Section 2
priority list.
