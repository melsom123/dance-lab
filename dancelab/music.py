"""Beat tracking with librosa, and lining movement up against the beat."""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np


@dataclass
class Beats:
    tempo: float
    times: np.ndarray          # beat times in seconds
    counts: np.ndarray         # dance count 1..8 for each beat (count 1 is a heuristic guess)
    onset_times: np.ndarray
    onset_env: np.ndarray


def extract_audio(video_path: str, sr: int = 22050) -> str | None:
    out = os.path.join(tempfile.mkdtemp(), "audio.wav")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), out],
                       capture_output=True)
    return out if r.returncode == 0 and os.path.getsize(out) > 1000 else None


def track_beats(video_path: str, start_bpm: float = 120.0) -> Beats | None:
    import librosa
    wav = extract_audio(video_path)
    if wav is None:
        return None
    y, sr = librosa.load(wav, sr=None)
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop,
                                                 start_bpm=start_bpm)
    times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    onset_times = librosa.times_like(onset_env, sr=sr, hop_length=hop)
    # Guess where "1" falls: the beat phase (mod 4) with the strongest average onset.
    strength = onset_env[beat_frames] if len(beat_frames) else np.array([])
    phase = int(np.argmax([strength[k::4].mean() if len(strength[k::4]) else 0 for k in range(4)])) if len(strength) else 0
    counts = ((np.arange(len(times)) - phase) % 8) + 1
    return Beats(float(np.atleast_1d(tempo)[0]), times, counts, onset_times, onset_env)


def hit_offsets(energy: np.ndarray, fps: float, beats: Beats) -> np.ndarray:
    """For each beat, offset (s) of the nearest movement-energy peak within half a beat.

    Negative = movement accent lands before the beat (rushing), positive = after (dragging).
    NaN where there is no clear peak.
    """
    from scipy.signal import find_peaks
    e = np.nan_to_num(energy)
    if len(beats.times) < 2:
        return np.array([])
    period = float(np.median(np.diff(beats.times)))
    peaks, _ = find_peaks(e, distance=max(1, int(0.25 * period * fps)), prominence=np.nanstd(e) * 0.3)
    peak_t = peaks / fps
    out = np.full(len(beats.times), np.nan)
    for i, bt in enumerate(beats.times):
        if len(peak_t) == 0:
            break
        j = np.argmin(np.abs(peak_t - bt))
        if abs(peak_t[j] - bt) <= period / 2:
            out[i] = peak_t[j] - bt
    return out
