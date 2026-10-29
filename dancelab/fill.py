"""Estimate a continuous full-body pose and splice it with the measured segments.

Measured frames are kept as they are. Everything else is estimated, and flagged:
- limbs out of frame in a usable shot: MediaPipe's own prediction for the hidden joint
- short gaps (cuts, dropouts): smooth interpolation between the measured poses either side
- long gaps (cutaways): motion borrowed from the nearest measured moment at the same point
  of the 8-count (whole 8-counts earlier or later), cross-faded into the neighbours.
  This assumes the routine repeats; it is a plausible fill, not a measurement.
- before the first and after the last measured frame nothing is estimated (NONE).

Poses are filled in a body-centred space (hip centre at the origin, units of torso length)
so cuts and zooms don't matter. Both 2D image poses (for drawing) and 3D world poses
(for angles) are filled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import pose as P
from .features import joint_ok, usable_mask

# Source of each frame's pose, per joint
MEASURED, MODEL_GUESS, INTERPOLATED, BORROWED, NONE = 0, 1, 2, 3, 4
SOURCE_NAMES = {MEASURED: "measured", MODEL_GUESS: "model guess", INTERPOLATED: "interpolated",
                BORROWED: "borrowed", NONE: "not estimated"}


@dataclass
class FilledPose:
    image_norm: np.ndarray   # (T, 33, 2) image coords, hip-centred, in torso lengths (y down)
    world: np.ndarray        # (T, 33, 3) world coords (metres, hip-centred)
    source: np.ndarray       # (T, 33) one of MEASURED / MODEL_GUESS / INTERPOLATED / BORROWED
    borrowed_from: np.ndarray  # (T,) source frame for BORROWED frames, -1 otherwise

    def frame_source(self, i: int) -> int:
        """Worst (most estimated) source among the body joints in frame i."""
        return int(self.source[i, 11:].max())

    @property
    def estimated(self) -> np.ndarray:
        return self.source[:, 11:].max(axis=1) != NONE


def _normalize_image(track: P.PoseTrack) -> np.ndarray:
    wh = np.array([track.width, track.height])
    img = track.image[:, :, :2] * wh
    hip = (img[:, P.L_HIP] + img[:, P.R_HIP]) / 2
    sh = (img[:, P.L_SHOULDER] + img[:, P.R_SHOULDER]) / 2
    torso = np.linalg.norm(sh - hip, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (img - hip[:, None]) / torso[:, None, None]


def _runs(mask: np.ndarray):
    t = 0
    while t < len(mask):
        if mask[t]:
            e = t
            while e < len(mask) and mask[e]:
                e += 1
            yield t, e
            t = e
        else:
            t += 1


def fill(track: P.PoseTrack, beats=None, max_interp_s: float = 1.5, blend_s: float = 0.25) -> FilledPose:
    T = len(track.image)
    usable = usable_mask(track)
    jok = joint_ok(track) & usable[:, None]
    img = _normalize_image(track)
    wld = track.world.copy()
    # Drop frames whose proportions are implausible (usually a guessed torso in a close-up):
    # any upper-arm, forearm, thigh or shin longer than 1.8 torso lengths.
    limbs = [(P.L_SHOULDER, P.L_ELBOW), (P.R_SHOULDER, P.R_ELBOW), (P.L_ELBOW, P.L_WRIST),
             (P.R_ELBOW, P.R_WRIST), (P.L_HIP, P.L_KNEE), (P.R_HIP, P.R_KNEE),
             (P.L_KNEE, P.L_ANKLE), (P.R_KNEE, P.R_ANKLE)]
    with np.errstate(invalid="ignore"):
        longest = np.max([np.linalg.norm(img[:, a] - img[:, b], axis=1) for a, b in limbs], axis=0)
        usable = usable & (longest < 1.8)
    jok &= usable[:, None]

    source = np.full((T, 33), INTERPOLATED, dtype=np.int8)
    source[usable] = MODEL_GUESS
    source[jok] = MEASURED
    img[~usable] = np.nan
    wld[~usable] = np.nan

    borrowed_from = np.full(T, -1)
    if not usable.any():
        return FilledPose(np.zeros((T, 33, 2)), np.zeros((T, 33, 3)), source, borrowed_from)

    good = np.flatnonzero(usable)
    period = float(np.median(np.diff(beats.times))) if beats is not None and len(beats.times) > 2 else None
    max_interp = int(max_interp_s * track.fps)
    blend = max(1, int(blend_s * track.fps))

    for a, b in _runs(~usable):
        if a == 0 or b == T:
            source[a:b] = NONE   # before the dance starts / after it ends
            continue
        if (b - a) <= max_interp or period is None:
            continue  # interpolated below
        # Borrow from whole 8-counts away, nearest measured frame first
        shift = int(round(8 * period * track.fps))
        for t in range(a, b):
            for n in sorted(range(-40, 41), key=abs):
                s = t + n * shift
                if n and 0 <= s < T and usable[s]:
                    img[t], wld[t] = img[s], wld[s]
                    source[t] = BORROWED
                    borrowed_from[t] = s
                    break
        # Cross-fade into the measured poses at the edges of the gap
        for edge, step in ((a, 1), (b - 1, -1)):
            nb = edge - step
            if not (0 <= nb < T and usable[nb]):
                continue
            for k in range(blend):
                t = edge + step * k
                if not (a <= t < b) or borrowed_from[t] < 0:
                    break
                wgt = (k + 1) / (blend + 1)   # weight of the borrowed pose
                img[t] = wgt * img[t] + (1 - wgt) * img[nb]
                wld[t] = wgt * wld[t] + (1 - wgt) * wld[nb]

    # Smooth interpolation through everything still missing (short gaps, and the ends)
    have = np.isfinite(img.reshape(T, -1)).all(axis=1) & np.isfinite(wld.reshape(T, -1)).all(axis=1)
    idx = np.flatnonzero(have)
    if len(idx) >= 2:
        miss = np.flatnonzero(~have)
        for arr in (img, wld):
            flat = arr.reshape(T, -1)
            flat[miss] = PchipInterpolator(idx, flat[idx], axis=0, extrapolate=False)(miss)
            flat[: idx[0]] = 0
            flat[idx[-1] + 1:] = 0
    return FilledPose(img, wld, source, borrowed_from)


def filled_track(track: P.PoseTrack, fp: FilledPose) -> P.PoseTrack:
    """A PoseTrack whose world coords are the filled ones and every joint counts as visible,
    so the usual angle code runs over the whole clip."""
    vis = np.ones((len(fp.world), 33))
    img = np.zeros_like(track.image)
    img[:, :, :2] = 0.5 + fp.image_norm * 0.1   # placeholder image coords, only used for hip sway
    return P.PoseTrack(track.fps, track.width, track.height, img, fp.world, vis)
