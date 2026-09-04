"""Chord estimation from chroma, plus a Lakh-MIDI symbolic sanity check.

The chord track is a *weak* symbolic signal layered on top of the audio graph,
not a transcription system. Template matching on chroma is deliberately simple.

**Read the caveat on :func:`validate_against_lmd` before quoting its number.**
Lakh Clean MIDI ships no audio, and this module does not synthesize any, so the
agreement figure is measured on MIDI-derived chroma. It bounds the template
matcher in isolation; it says nothing about chord estimation from real
recordings, where polyphony, percussion, reverb and mastering all degrade chroma.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .utils import get_logger, resolve_path

LOGGER = get_logger("gbmc.chords")

__all__ = [
    "CHORD_VOCAB",
    "PITCH_CLASSES",
    "chord_templates",
    "estimate_chords",
    "chords_from_midi",
    "validate_against_lmd",
    "chord_transition_matrix",
    "LMD_VALIDATION_CAVEAT",
    "LMD_VALIDATION_METHODOLOGY",
]

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

#: 12 major + 12 minor triads plus the no-chord symbol -- 25 entries, matching
#: ``config.graph.chord_vocab_size``.
CHORD_VOCAB: list[str] = (
    [f"{p}:maj" for p in PITCH_CLASSES] + [f"{p}:min" for p in PITCH_CLASSES] + ["N"]
)

_MAJ_INTERVALS = (0, 4, 7)
_MIN_INTERVALS = (0, 3, 7)


def chord_templates() -> np.ndarray:
    """``[24, 12]`` binary triad templates, L2-normalised, in CHORD_VOCAB order."""
    templates = np.zeros((24, 12), dtype=np.float64)
    for root in range(12):
        for interval in _MAJ_INTERVALS:
            templates[root, (root + interval) % 12] = 1.0
        for interval in _MIN_INTERVALS:
            templates[12 + root, (root + interval) % 12] = 1.0
    return templates / np.linalg.norm(templates, axis=1, keepdims=True)


_TEMPLATES = chord_templates()


def _median_smooth(labels: Sequence[int], width: int) -> list[int]:
    """Mode filter over a label sequence (median is meaningless on symbols).

    Chroma frames are ~23 ms; without smoothing the estimator flips chords
    several times per beat, which produces a transition matrix dominated by
    noise rather than by harmony.
    """
    labels = list(labels)
    width = int(width)
    if width <= 1 or len(labels) <= 2:
        return labels
    if width % 2 == 0:
        width += 1
    half = width // 2
    out = []
    for i in range(len(labels)):
        window = labels[max(0, i - half): i + half + 1]
        values, counts = np.unique(window, return_counts=True)
        best = values[np.argmax(counts)]
        # prefer keeping the current label when the window is tied
        if counts.max() == np.sum(np.asarray(window) == labels[i]):
            best = labels[i]
        out.append(int(best))
    return out


def estimate_chords(chroma, cfg=None) -> list[str]:
    """Frame-wise chord labels from a ``[12, T]`` chroma matrix.

    Each frame is L2-normalised and matched against the 24 triad templates by
    cosine similarity; frames whose total energy falls below ``chords.min_energy``
    are labelled ``"N"``. A mode filter then removes single-frame flicker.
    """
    chroma = np.asarray(chroma, dtype=np.float64)
    if chroma.ndim != 2:
        raise ValueError(f"chroma must be [12, T], got shape {chroma.shape}")
    if chroma.shape[0] != 12 and chroma.shape[1] == 12:
        chroma = chroma.T
    if chroma.shape[0] != 12:
        raise ValueError(f"chroma must have 12 pitch classes, got {chroma.shape[0]}")

    cfg = cfg or {}
    chord_cfg = cfg.get("chords", {}) if isinstance(cfg, dict) else {}
    min_energy = float(chord_cfg.get("min_energy", 1e-6))
    width = int(chord_cfg.get("smoothing_frames", 9))

    energy = chroma.sum(axis=0)
    norms = np.linalg.norm(chroma, axis=0)
    safe = np.maximum(norms, 1e-12)
    unit = chroma / safe[None, :]
    scores = _TEMPLATES @ unit                    # [24, T]
    labels = np.argmax(scores, axis=0).astype(int)
    labels[(energy < min_energy) | (norms < 1e-12)] = 24   # "N"

    labels = _median_smooth(labels, width)
    return [CHORD_VOCAB[i] for i in labels]


def chords_from_midi(midi_path, fs: float = 10.0) -> list[str]:
    """Ground-truth-ish chord labels from a MIDI file via pretty_midi chroma.

    Sampled at ``fs`` Hz. Drum tracks are excluded -- a kick drum contributes
    broadband energy that the chroma folds into an arbitrary pitch class.
    """
    import pretty_midi

    path = str(resolve_path(midi_path))
    midi = pretty_midi.PrettyMIDI(path)
    pitched = [inst for inst in midi.instruments if not inst.is_drum]
    if not pitched:
        return []

    end = midi.get_end_time()
    if end <= 0:
        return []
    times = np.arange(0, end, 1.0 / fs)
    chroma = np.zeros((12, times.size), dtype=np.float64)
    for inst in pitched:
        for note in inst.notes:
            lo = int(np.searchsorted(times, note.start, side="left"))
            hi = int(np.searchsorted(times, note.end, side="right"))
            if hi > lo:
                chroma[note.pitch % 12, lo:hi] += note.velocity / 127.0

    labels = []
    for t in range(chroma.shape[1]):
        column = chroma[:, t]
        if column.sum() <= 0:
            labels.append("N")
            continue
        unit = column / np.linalg.norm(column)
        labels.append(CHORD_VOCAB[int(np.argmax(_TEMPLATES @ unit))])
    return labels


#: Stored alongside every result so the number cannot be quoted without it.
LMD_VALIDATION_METHODOLOGY = "midi_derived_chroma"
LMD_VALIDATION_CAVEAT = (
    "Lakh Clean MIDI ships no audio and none is synthesized here, so both sides "
    "of this comparison derive from the same MIDI note events: the 'input' is "
    "pretty_midi.get_chroma and the 'ground truth' is the same notes read "
    "directly. This therefore does NOT validate chord estimation from audio. It "
    "is an upper bound on the template-matching and smoothing step in isolation, "
    "measured on noiseless, perfectly-separated chroma. Real MTAT/FMA mp3s add "
    "polyphony, percussion, reverb, overtones and mastering, all of which smear "
    "chroma; expect substantially lower agreement there. No audio-domain chord "
    "accuracy is claimed anywhere in this project."
)


def validate_against_lmd(lmd_dir, n_samples: int = 200, cfg=None, seed: int = 42) -> dict:
    """Frame-level agreement between template matching and MIDI note labels.

    .. warning::

       **This does not validate the audio chord pipeline.** For each sampled
       file we take ``pretty_midi.get_chroma`` as the "audio-like" input and
       compare against note-level labels read from the *same* MIDI. No audio is
       synthesized (Lakh Clean MIDI ships none), so the chroma is noiseless and
       perfectly source-separated. The result is an upper bound on the template
       matching + median smoothing step alone. See
       :data:`LMD_VALIDATION_CAVEAT`, which is copied into every returned dict
       and into the persisted JSON so the number travels with its limitation.

    Reports overall agreement, root-only agreement (ignoring the maj/min
    decision, which chroma genuinely struggles with when a third is absent), and
    the per-file spread.
    """
    root = resolve_path(lmd_dir)
    result = {
        "lmd_dir": str(root),
        "methodology": LMD_VALIDATION_METHODOLOGY,
        "audio_synthesized": False,
        "validates_audio_pipeline": False,
        "caveat": LMD_VALIDATION_CAVEAT,
        "n_requested": int(n_samples),
        "n_files_found": 0,
        "n_files_scored": 0,
        "frame_agreement": float("nan"),
        "root_agreement": float("nan"),
        "per_file_agreement_std": float("nan"),
        "errors": 0,
    }
    if not root.exists():
        LOGGER.warning("Lakh MIDI directory not found: %s", root)
        result["status"] = "missing"
        return result

    files = sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi"))
    result["n_files_found"] = len(files)
    if not files:
        result["status"] = "empty"
        return result

    rng = np.random.default_rng(seed)
    picks = rng.permutation(len(files))[: int(n_samples)]

    per_file, total_match, total_frames, root_match = [], 0, 0, 0
    for i in picks:
        path = files[int(i)]
        try:
            import pretty_midi

            midi = pretty_midi.PrettyMIDI(str(path))
            chroma = midi.get_chroma(fs=10.0)
            truth = chords_from_midi(path, fs=10.0)
        except Exception:
            result["errors"] += 1
            continue
        if chroma.size == 0 or not truth:
            result["errors"] += 1
            continue

        pred = estimate_chords(chroma, cfg)
        n = min(len(pred), len(truth))
        if n == 0:
            continue
        pred, truth = pred[:n], truth[:n]
        matches = sum(p == t for p, t in zip(pred, truth))
        roots = sum(p.split(":")[0] == t.split(":")[0] for p, t in zip(pred, truth))
        per_file.append(matches / n)
        total_match += matches
        root_match += roots
        total_frames += n
        result["n_files_scored"] += 1

    if total_frames:
        result["frame_agreement"] = float(total_match / total_frames)
        result["root_agreement"] = float(root_match / total_frames)
        result["per_file_agreement_std"] = float(np.std(per_file)) if per_file else 0.0
        result["n_frames"] = int(total_frames)
    result["status"] = "ok" if result["n_files_scored"] else "no_scorable_files"
    if result["n_files_scored"]:
        LOGGER.warning(
            "chord agreement %.3f measured on MIDI-DERIVED chroma, not audio. "
            "Upper bound on the template matcher alone; do not report it as an "
            "audio chord accuracy.", result["frame_agreement"],
        )
    return result


def chord_transition_matrix(seq: Iterable[str], vocab: Sequence[str] | None = None) -> np.ndarray:
    """Raw transition counts ``[V, V]`` for a chord sequence.

    Counts, not probabilities: the graph builder row-normalises when it needs a
    distribution, and analysis code sometimes wants the raw support.
    Consecutive repeats of the same symbol are collapsed, so the diagonal
    reflects genuine chord re-attacks rather than the frame rate.
    """
    vocab = list(vocab) if vocab is not None else list(CHORD_VOCAB)
    index = {name: i for i, name in enumerate(vocab)}
    matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float64)

    collapsed: list[str] = []
    for symbol in seq:
        symbol = str(symbol)
        if symbol not in index:
            continue
        if not collapsed or collapsed[-1] != symbol:
            collapsed.append(symbol)

    for a, b in zip(collapsed[:-1], collapsed[1:]):
        matrix[index[a], index[b]] += 1.0
    if len(collapsed) == 1:
        matrix[index[collapsed[0]], index[collapsed[0]]] = 1.0
    return matrix


def chords_to_segment_map(chord_seq: Sequence[str], n_segments: int,
                          present: Sequence[str] | None = None) -> list[int]:
    """Map each segment to the chord node that dominates its time span."""
    if n_segments <= 0 or not len(chord_seq):
        return []
    present = list(present) if present is not None else sorted(set(chord_seq))
    local = {name: i for i, name in enumerate(present)}
    bounds = np.linspace(0, len(chord_seq), n_segments + 1).round().astype(int)
    out = []
    for s in range(n_segments):
        lo, hi = bounds[s], max(bounds[s + 1], bounds[s] + 1)
        window = [c for c in chord_seq[lo:hi] if c in local]
        if not window:
            out.append(local.get("N", 0))
            continue
        values, counts = np.unique(window, return_counts=True)
        out.append(local[str(values[np.argmax(counts)])])
    return out


def main(argv=None) -> int:
    """CLI: validate the chroma chord estimator against Lakh MIDI ground truth."""
    import argparse
    import json

    from .utils import load_config, save_json

    parser = argparse.ArgumentParser(
        description="Frame-level agreement between template matching and MIDI."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--lmd-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    lmd_dir = args.lmd_dir or cfg["datasets"]["lmd"]["root"]
    report = validate_against_lmd(lmd_dir, n_samples=args.n_samples, cfg=cfg,
                                  seed=args.seed)
    save_json(report, resolve_path(cfg["paths"]["results"]) / "chord_validation.json")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
