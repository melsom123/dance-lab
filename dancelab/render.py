"""Overlay videos and plots."""
from __future__ import annotations

import os
import subprocess

import cv2
import numpy as np

from . import pose as P

LEFT_COLOR = (255, 160, 0)    # BGR: blue-ish for the dancer's left
RIGHT_COLOR = (0, 140, 255)   # orange for the dancer's right
GHOST_COLOR = (80, 255, 80)


def draw_skeleton(frame, pts_px, color_left=LEFT_COLOR, color_right=RIGHT_COLOR, thickness=3, alpha=1.0):
    if pts_px is None or np.isnan(pts_px).any():
        return frame
    layer = frame.copy() if alpha < 1 else frame
    for a, b in P.EDGES:
        col = color_left if (a in P.LEFT_SIDE and b in P.LEFT_SIDE) else (
            color_right if (a not in P.LEFT_SIDE and b not in P.LEFT_SIDE) else (200, 200, 200))
        cv2.line(layer, tuple(map(int, pts_px[a])), tuple(map(int, pts_px[b])), col, thickness, cv2.LINE_AA)
    for i in range(11, 33):
        cv2.circle(layer, tuple(map(int, pts_px[i])), thickness + 1, (255, 255, 255), -1, cv2.LINE_AA)
    if alpha < 1:
        cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0, dst=frame)
    return frame


def _text(img, s, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def mux_audio(silent_video: str, audio_src: str, out_path: str, audio_offset_s: float = 0.0):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", silent_video]
    if audio_offset_s:
        cmd += ["-ss", f"{audio_offset_s:.3f}"]
    cmd += ["-i", audio_src, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", "-preset", "veryfast", "-c:a", "aac", "-shortest", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:  # fall back to re-encoded video without audio
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent_video, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", out_path], check=True)


def overlay_video(video_path, track: P.PoseTrack, angles: dict, energy, beats, out_path,
                  show_angles=("l_elbow", "r_elbow", "l_knee", "r_knee", "torso_lean")):
    cap = cv2.VideoCapture(video_path)
    fps, w, h = track.fps, track.width, track.height
    tmp = out_path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    beat_t = beats.times if beats else np.array([])
    e_max = np.nanpercentile(energy, 98) if np.isfinite(energy).any() else 1
    strip_h, strip_s = max(60, h // 8), 4.0   # energy strip: last/next 2 s
    for i in range(len(track.image)):
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        pts = track.image[i, :, :2] * np.array([w, h])
        draw_skeleton(frame, pts)

        # Beat flash + count
        if len(beat_t):
            k = int(np.argmin(np.abs(beat_t - t)))
            dt = t - beat_t[k]
            if 0 <= dt < 0.12:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 255), max(6, w // 100))
            last = k if beat_t[k] <= t else k - 1
            if last >= 0:
                _text(frame, str(beats.counts[last]), (w - 90, 70), 2.0, (0, 255, 255), 4)
            _text(frame, f"{beats.tempo:.0f} BPM", (w - 160, 105), 0.6)

        # Angle readouts
        for j, name in enumerate(show_angles):
            v = angles[name][i]
            if np.isfinite(v):
                _text(frame, f"{name:<11}{v:6.0f}", (12, 28 + 24 * j), 0.6)

        # Energy strip with beat ticks; the centre line is "now"
        y0 = h - strip_h
        roi = frame[y0:h]
        roi[:] = (roi * 0.35).astype(np.uint8)
        x_of = lambda tt: int((tt - t + strip_s / 2) / strip_s * w)
        for bt, c in zip(beat_t, beats.counts if beats else []):
            if abs(bt - t) < strip_s / 2:
                x = x_of(bt)
                cv2.line(frame, (x, y0), (x, h), (0, 200, 255) if c in (1, 5) else (0, 120, 160), 2 if c in (1, 5) else 1)
        i0, i1 = max(0, int((t - strip_s / 2) * fps)), min(len(energy), int((t + strip_s / 2) * fps))
        seg = [(x_of(ii / fps), int(h - 4 - (strip_h - 8) * min(1, np.nan_to_num(energy[ii]) / e_max)))
               for ii in range(i0, i1)]
        if len(seg) > 1:
            cv2.polylines(frame, [np.array(seg, np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (w // 2, y0), (w // 2, h), (0, 0, 255), 1)
        _text(frame, "movement energy", (8, y0 + 18), 0.5)
        vw.write(frame)
    cap.release()
    vw.release()
    mux_audio(tmp, video_path, out_path)
    os.remove(tmp)


def plot_angles(angles: dict, energy, fps, beats, out_png, hits=None, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(energy)) / fps
    groups = [("Arms", ["l_elbow", "r_elbow", "l_shoulder", "r_shoulder"]),
              ("Legs", ["l_knee", "r_knee", "l_hip", "r_hip"]),
              ("Torso", ["torso_lean", "torso_twist"]),
              ("Hips (torso lengths)", ["hip_sway", "hip_bounce"])]
    fig, axes = plt.subplots(len(groups) + 1, 1, figsize=(14, 2.4 * (len(groups) + 1)), sharex=True)
    for ax, (label, names) in zip(axes, groups):
        for n in names:
            ax.plot(t, angles[n], lw=1.2, label=n, ls="-" if n.startswith(("l_", "torso", "hip_sway")) else "--")
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=7, ncol=4)
    axes[-1].plot(t, energy, color="k", lw=1)
    axes[-1].set_ylabel("Movement\nenergy")
    axes[-1].set_xlabel("time (s)")
    if beats is not None:
        for ax in axes:
            for bt, c in zip(beats.times, beats.counts):
                ax.axvline(bt, color="tab:red" if c == 1 else "0.75", lw=1.2 if c == 1 else 0.6, zorder=0)
    if hits is not None and len(hits):
        ok = np.isfinite(hits)
        axes[-1].scatter(beats.times[ok] + hits[ok], np.interp(beats.times[ok] + hits[ok], t, np.nan_to_num(energy)),
                         color="tab:red", s=12, zorder=3, label="accent nearest each beat")
        axes[-1].legend(loc="upper right", fontsize=7)
    fig.suptitle(title or "Joint angles vs. beat (red line = estimated count 1)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
