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

    # Hip sway / bounce in the image, in torso lengths (camera-scale independent)
    img = smooth(track.image[:, :, :2], track.fps)
    hip_img = (img[:, P.L_HIP] + img[:, P.R_HIP]) / 2
    sh_img = (img[:, P.L_SHOULDER] + img[:, P.R_SHOULDER]) / 2
    aspect = np.array([track.width, track.height])
    torso_px = np.nanmedian(np.linalg.norm((sh_img - hip_img) * aspect, axis=1))
    hip_px = hip_img * aspect
    out["hip_sway"] = (hip_px[:, 0] - np.nanmean(hip_px[:, 0])) / torso_px
    out["hip_bounce"] = -(hip_px[:, 1] - np.nanmean(hip_px[:, 1])) / torso_px  # up is positive
    return out


def motion_energy(track: P.PoseTrack) -> np.ndarray:
    """Total joint speed per frame (m/s summed over limbs), a proxy for movement accents."""
    w = smooth(track.world, track.fps)
    joints = [P.L_ELBOW, P.R_ELBOW, P.L_WRIST, P.R_WRIST, P.L_KNEE, P.R_KNEE,
              P.L_ANKLE, P.R_ANKLE, P.L_HIP, P.R_HIP, P.L_SHOULDER, P.R_SHOULDER]
    v = np.linalg.norm(np.gradient(w[:, joints], axis=0), axis=-1) * track.fps
    e = np.nansum(v, axis=1)
    e = smooth(e[:, None], track.fps, 0.12)[:, 0]
    e[np.isnan(w[:, 0, 0])] = np.nan
    return e


def body_scale(track: P.PoseTrack) -> float:
    """Median shoulder-to-hip distance in pixels, for normalizing overlays."""
    img = track.image[:, :, :2] * np.array([track.width, track.height])
    sh = (img[:, P.L_SHOULDER] + img[:, P.R_SHOULDER]) / 2
    hp = (img[:, P.L_HIP] + img[:, P.R_HIP]) / 2
    return float(np.nanmedian(np.linalg.norm(sh - hp, axis=1)))
