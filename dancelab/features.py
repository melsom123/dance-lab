"""Joint angles and movement features from a PoseTrack."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from . import pose as P

# name -> (a, b, c): angle at b between b->a and b->c, in degrees, from 3D world coords
TRIPLETS = {
    "l_elbow": (P.L_SHOULDER, P.L_ELBOW, P.L_WRIST),
    "r_elbow": (P.R_SHOULDER, P.R_ELBOW, P.R_WRIST),
    "l_shoulder": (P.L_HIP, P.L_SHOULDER, P.L_ELBOW),   # arm raise relative to torso
    "r_shoulder": (P.R_HIP, P.R_SHOULDER, P.R_ELBOW),
    "l_hip": (P.L_SHOULDER, P.L_HIP, P.L_KNEE),
    "r_hip": (P.R_SHOULDER, P.R_HIP, P.R_KNEE),
    "l_knee": (P.L_HIP, P.L_KNEE, P.L_ANKLE),
    "r_knee": (P.R_HIP, P.R_KNEE, P.R_ANKLE),
    "l_ankle": (P.L_KNEE, P.L_ANKLE, P.L_FOOT),
    "r_ankle": (P.R_KNEE, P.R_ANKLE, P.R_FOOT),
}


BODY = [P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP, P.L_KNEE, P.R_KNEE, P.L_ANKLE, P.R_ANKLE]


def usable_mask(track: P.PoseTrack, min_vis: float = 0.6, min_run_s: float = 1.0) -> np.ndarray:
    """Frames showing a full-body dancer: body landmarks confidently visible and inside the frame.

    Filters out cutaways (audience shots, close-ups with legs out of frame) where MediaPipe
    hallucinates a skeleton. Runs shorter than min_run_s are dropped as flicker.
    """
    vis = np.nanmean(track.vis[:, BODY], axis=1)
    xy = track.image[:, BODY, :2]
    with np.errstate(invalid="ignore"):
        inside = np.all((xy > -0.02) & (xy < 1.02), axis=(1, 2))
        ok = (vis >= min_vis) & inside
    min_run = int(min_run_s * track.fps)
    t = 0
    while t < len(ok):
        if ok[t]:
            e = t
            while e < len(ok) and ok[e]:
                e += 1
            if e - t < min_run:
                ok[t:e] = False
            t = e
        else:
            t += 1
    return ok


def usable_segments(mask: np.ndarray, fps: float) -> list[tuple[float, float]]:
    segs, t = [], 0
    while t < len(mask):
        if mask[t]:
            e = t
            while e < len(mask) and mask[e]:
                e += 1
            segs.append((t / fps, e / fps))
            t = e
        else:
            t += 1
    return segs


def smooth(x: np.ndarray, fps: float, window_s: float = 0.15) -> np.ndarray:
    """Savitzky-Golay smoothing along time, skipping NaN rows."""
    win = max(5, int(window_s * fps) | 1)
    out = x.copy()
    good = ~np.isnan(x.reshape(len(x), -1)).any(axis=1)
    if good.sum() > win:
        out[good] = savgol_filter(x[good], win, 2, axis=0)
    return out


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def joint_angles(track: P.PoseTrack) -> dict[str, np.ndarray]:
    w = smooth(track.world, track.fps)
    out = {name: _angle(w[:, a], w[:, b], w[:, c]) for name, (a, b, c) in TRIPLETS.items()}

    mid_sh = (w[:, P.L_SHOULDER] + w[:, P.R_SHOULDER]) / 2
    mid_hip = (w[:, P.L_HIP] + w[:, P.R_HIP]) / 2
    spine = mid_sh - mid_hip
    # Lean: angle of spine away from vertical (world y points down, so "up" is -y)
    up = np.array([0, -1, 0])
    out["torso_lean"] = np.degrees(np.arccos(np.clip(spine @ up / (np.linalg.norm(spine, axis=1) + 1e-9), -1, 1)))
    # Twist: shoulder line vs hip line, projected on the floor plane (x, z)
    sh = (w[:, P.R_SHOULDER] - w[:, P.L_SHOULDER])[:, [0, 2]]
    hp = (w[:, P.R_HIP] - w[:, P.L_HIP])[:, [0, 2]]
    twist = np.degrees(np.arctan2(sh[:, 1], sh[:, 0]) - np.arctan2(hp[:, 1], hp[:, 0]))
    out["torso_twist"] = (twist + 180) % 360 - 180

    # Hip sway / bounce in the image, relative to a 2 s moving average and measured in
    # torso lengths, so camera zoom, cuts and travelling across the stage don't dominate.
    img = smooth(track.image[:, :, :2], track.fps)
    aspect = np.array([track.width, track.height])
    hip_px = (img[:, P.L_HIP] + img[:, P.R_HIP]) / 2 * aspect
    sh_px = (img[:, P.L_SHOULDER] + img[:, P.R_SHOULDER]) / 2 * aspect
    torso_px = np.linalg.norm(sh_px - hip_px, axis=1)
    ok = usable_mask(track) & np.isfinite(torso_px)
    win = int(2 * track.fps)
    hip_rel = np.full_like(hip_px, np.nan)
    for t in np.flatnonzero(ok):
        lo, hi = max(0, t - win // 2), min(len(ok), t + win // 2 + 1)
        sel = ok[lo:hi]
        hip_rel[t] = (hip_px[t] - hip_px[lo:hi][sel].mean(axis=0)) / torso_px[t]
    out["hip_sway"] = hip_rel[:, 0]
    out["hip_bounce"] = -hip_rel[:, 1]  # up is positive
    bad = ~usable_mask(track)
    for v in out.values():
        v[bad] = np.nan
    return out


def motion_energy(track: P.PoseTrack) -> np.ndarray:
    """Total joint speed per frame (m/s summed over limbs), a proxy for movement accents."""
    w = smooth(track.world, track.fps)
    joints = [P.L_ELBOW, P.R_ELBOW, P.L_WRIST, P.R_WRIST, P.L_KNEE, P.R_KNEE,
              P.L_ANKLE, P.R_ANKLE, P.L_HIP, P.R_HIP, P.L_SHOULDER, P.R_SHOULDER]
    v = np.linalg.norm(np.gradient(w[:, joints], axis=0), axis=-1) * track.fps
    e = np.nansum(v, axis=1)
    e = smooth(e[:, None], track.fps, 0.12)[:, 0]
    e[np.isnan(w[:, 0, 0]) | ~usable_mask(track)] = np.nan
    return e


def body_scale(track: P.PoseTrack) -> float:
    """Median shoulder-to-hip distance in pixels, for normalizing overlays."""
    img = track.image[:, :, :2] * np.array([track.width, track.height])
    sh = (img[:, P.L_SHOULDER] + img[:, P.R_SHOULDER]) / 2
    hp = (img[:, P.L_HIP] + img[:, P.R_HIP]) / 2
    return float(np.nanmedian(np.linalg.norm(sh - hp, axis=1)))
