"""Compare your recording against a reference: DTW time alignment, angle errors, ghost overlay."""
from __future__ import annotations

import os

import cv2
import numpy as np

from . import pose as P
from .features import joint_angles, joint_ok
from .render import GHOST_COLOR, _text, draw_skeleton, mux_audio

COMPARE_JOINTS = ["l_elbow", "r_elbow", "l_shoulder", "r_shoulder", "l_hip", "r_hip",
                  "l_knee", "r_knee", "torso_lean", "torso_twist"]

# Index permutation that swaps left and right landmarks (for mirror-image practice)
MIRROR_IDX = list(range(33))
for a, b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)] + [(i, i + 1) for i in range(11, 33, 2)]:
    MIRROR_IDX[a], MIRROR_IDX[b] = b, a


def mirror_track(t: P.PoseTrack) -> P.PoseTrack:
    img = t.image[:, MIRROR_IDX].copy(); img[..., 0] = 1 - img[..., 0]
    wld = t.world[:, MIRROR_IDX].copy(); wld[..., 0] = -wld[..., 0]
    return P.PoseTrack(t.fps, t.width, t.height, img, wld, t.vis[:, MIRROR_IDX], t.people, t.solo_only)


def flip_track(t: P.PoseTrack) -> P.PoseTrack:
    """Horizontally flip the picture without relabelling sides: the dancer's right arm is still
    labelled right, but now appears on the right of the screen. This is what a "mirrored"
    tutorial video is: facing the screen, you use the same arm as the side of the screen it's on."""
    img = t.image.copy(); img[..., 0] = 1 - img[..., 0]
    wld = t.world.copy(); wld[..., 0] = -wld[..., 0]
    return P.PoseTrack(t.fps, t.width, t.height, img, wld, t.vis, t.people, t.solo_only)


def _feature_matrix(angles: dict) -> np.ndarray:
    X = np.stack([angles[j] for j in COMPARE_JOINTS], axis=0) / 45.0   # degrees -> ~unit scale
    # Fill NaNs so DTW doesn't break
    for r in X:
        bad = ~np.isfinite(r)
        if bad.all():
            r[:] = 0
        elif bad.any():
            r[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), r[~bad])
    return X


def align(ref: P.PoseTrack, you: P.PoseTrack, ref_angles: dict, you_angles: dict):
    """DTW alignment. Returns (map ref frame -> your frame, per-frame per-joint abs error in degrees)."""
    import librosa
    # Resample your track to the reference fps so the band constraint is meaningful
    Xr, Xy = _feature_matrix(ref_angles), _feature_matrix(you_angles)
    _, wp = librosa.sequence.dtw(X=Xr, Y=Xy, metric="euclidean", backtrack=True)
    wp = wp[::-1]
    ref_to_you = np.zeros(Xr.shape[1], dtype=int)
    for r, y in wp:
        ref_to_you[r] = y   # last match wins; path is monotonic so this is fine
    err = {j: np.abs(ref_angles[j] - you_angles[j][ref_to_you]) for j in COMPARE_JOINTS}
    for j in ("torso_twist",):
        err[j] = np.abs((ref_angles[j] - you_angles[j][ref_to_you] + 180) % 360 - 180)
    return ref_to_you, err


def _ghost_points(ref_pts, you_pts):
    """Move/scale the reference skeleton onto your body (hip centre + torso length)."""
    def anchor(p):
        hip = (p[P.L_HIP] + p[P.R_HIP]) / 2
        sh = (p[P.L_SHOULDER] + p[P.R_SHOULDER]) / 2
        return hip, np.linalg.norm(sh - hip)
    rh, rs = anchor(ref_pts)
    yh, ys = anchor(you_pts)
    if not (np.isfinite(rs) and np.isfinite(ys)) or rs < 1:
        return None
    return (ref_pts - rh) * (ys / rs) + yh


def comparison_video(ref_video, you_video, ref: P.PoseTrack, you: P.PoseTrack, ref_to_you, err, out_path,
                     beats=None, panel_h=540, mirror=False):
    rc, yc = cv2.VideoCapture(ref_video), cv2.VideoCapture(you_video)
    def read_all(cap):
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        return frames
    you_frames = read_all(yc)
    rw = int(ref.width * panel_h / ref.height)
    yw = int(you.width * panel_h / you.height)
    tmp = out_path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), ref.fps, (rw + yw, panel_h))
    rok, yok = joint_ok(ref), joint_ok(you)
    total_err = np.nanmean(np.stack([err[j] for j in COMPARE_JOINTS]), axis=0)
    worst = lambda i: sorted(COMPARE_JOINTS, key=lambda j: -np.nan_to_num(err[j][i]))[:3]
    for i in range(len(ref_to_you)):
        ok, rf = rc.read()
        if not ok:
            break
        j = min(ref_to_you[i], len(you_frames) - 1)
        yf = you_frames[j].copy()
        rpts = ref.image[i, :, :2] * [ref.width, ref.height]
        ypts = you.image[j, :, :2] * [you.width, you.height]
        draw_skeleton(rf, rpts, ok=rok[i])
        draw_skeleton(yf, ypts, ok=yok[j])
        gref, gok = rpts, rok[i]
        if mirror:
            gok = rok[i][MIRROR_IDX]  # flip the reference about its own hip centre so it matches your mirror-image dancing
            gref = rpts[MIRROR_IDX].copy()
            gref[:, 0] = (rpts[P.L_HIP, 0] + rpts[P.R_HIP, 0]) - gref[:, 0]
        g = _ghost_points(gref, ypts) if np.isfinite(rpts[P.L_SHOULDER:P.R_HIP + 1]).all() and np.isfinite(ypts[P.L_SHOULDER:P.R_HIP + 1]).all() else None
        if g is not None:
            draw_skeleton(yf, g, GHOST_COLOR, GHOST_COLOR, thickness=2, alpha=0.7, ok=gok)
        rf = cv2.resize(rf, (rw, panel_h)); yf = cv2.resize(yf, (yw, panel_h))
        _text(rf, "REFERENCE", (10, 25), 0.7)
        _text(yf, "YOU (time-aligned)  green = reference pose", (10, 25), 0.6)
        if np.isfinite(total_err[i]):
            col = (0, 220, 0) if total_err[i] < 15 else (0, 200, 255) if total_err[i] < 30 else (0, 0, 255)
            _text(yf, f"avg angle diff {total_err[i]:4.0f} deg", (10, 55), 0.6, col)
            for k, name in enumerate(worst(i)):
                _text(yf, f"{name}: {err[name][i]:.0f}", (10, 80 + 22 * k), 0.5)
        if beats is not None and len(beats.times):
            t = i / ref.fps
            last = np.searchsorted(beats.times, t, side="right") - 1
            if last >= 0:
                _text(rf, str(beats.counts[last]), (rw - 60, 50), 1.5, (0, 255, 255), 3)
        vw.write(np.hstack([rf, yf]))
    rc.release(); vw.release()
    mux_audio(tmp, ref_video, out_path)
    os.remove(tmp)


def summary(err, ref_fps, beats=None) -> str:
    lines = ["Mean absolute angle difference after time alignment (degrees):"]
    for j in sorted(COMPARE_JOINTS, key=lambda j: -np.nanmean(err[j])):
        lines.append(f"  {j:<12} {np.nanmean(err[j]):6.1f}")
    if beats is not None and len(beats.times):
        total = np.nanmean(np.stack([err[j] for j in COMPARE_JOINTS]), axis=0)
        per_beat = [(k, np.nanmean(total[int(a * ref_fps):int(b * ref_fps)]))
                    for k, (a, b) in enumerate(zip(beats.times[:-1], beats.times[1:]))]
        per_beat = [x for x in per_beat if np.isfinite(x[1])]
        lines.append("Beats where you differ most (beat #, count, avg deg):")
        for k, v in sorted(per_beat, key=lambda x: -x[1])[:8]:
            lines.append(f"  beat {k:3d}  count {beats.counts[k]}  at {beats.times[k]:6.2f}s  {v:5.1f}")
    return "\n".join(lines)
