"""Practice material: looped, slowed, mirrored clips per 8-count, and a printable count sheet."""
from __future__ import annotations

import os
import subprocess

import numpy as np

from . import pose as P
from .features import joint_ok, usable_mask

CUE_JOINTS = {
    # name: (body part label, phrase when the angle decreases, phrase when it increases)
    "l_elbow": ("{L} elbow", "bends", "straightens"),
    "r_elbow": ("{R} elbow", "bends", "straightens"),
    "l_shoulder": ("{L} arm", "drops", "lifts"),
    "r_shoulder": ("{R} arm", "drops", "lifts"),
    "l_knee": ("{L} knee", "bends", "straightens"),
    "r_knee": ("{R} knee", "bends", "straightens"),
    "l_hip": ("{L} leg", "lifts", "lowers"),
    "r_hip": ("{R} leg", "lifts", "lowers"),
    "torso_lean": ("torso", "straightens up", "leans"),
}


def eight_counts(beats) -> list[tuple[int, int]]:
    """(first beat index, end beat index) for each 8-count starting on an estimated count 1."""
    ones = [i for i, c in enumerate(beats.counts) if c == 1]
    return [(i, min(i + 8, len(beats.times) - 1)) for i in ones if i + 1 < len(beats.times)]


def cue_text(prev: dict, cur: dict, min_change: float = 25.0, max_cues: int = 2) -> str:
    """Describe the biggest joint changes from the previous count, in plain words."""
    L, R = "left", "right"
    changes = []
    for name, (part, dec, inc) in CUE_JOINTS.items():
        a, b = prev.get(name), cur.get(name)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        d = b - a
        if abs(d) >= min_change:
            changes.append((abs(d), f"{part.format(L=L, R=R)} {inc if d > 0 else dec}"))
    tw_a, tw_b = prev.get("torso_twist"), cur.get("torso_twist")
    if tw_a is not None and tw_b is not None and np.isfinite(tw_a) and np.isfinite(tw_b) and abs(tw_b - tw_a) >= min_change:
        changes.append((abs(tw_b - tw_a), "twist"))
    if not changes:
        return "hold / small move"
    changes.sort(reverse=True)
    return ", ".join(c for _, c in changes[:max_cues])


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def practice_clips(rendered_video: str, beats, outdir: str, speeds=(0.5, 0.75, 1.0), loops: int = 4,
                   lead_beats: int = 4) -> list[str]:
    """Cut each 8-count (plus a lead-in) from an already rendered video, slow it and loop it."""
    os.makedirs(outdir, exist_ok=True)
    out = []
    period = float(np.median(np.diff(beats.times)))
    for n, (b0, b1) in enumerate(eight_counts(beats), start=1):
        start = max(0.0, beats.times[b0] - lead_beats * period)
        end = beats.times[b1] if b1 > b0 + 7 else beats.times[b0] + 8 * period
        seg = os.path.join(outdir, f"_seg{n:02d}.mp4")
        _ffmpeg("-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", rendered_video,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", seg)
        for sp in speeds:
            path = os.path.join(outdir, f"8count_{n:02d}_{sp:.2f}x.mp4")
            _ffmpeg("-stream_loop", str(loops - 1), "-i", seg,
                    "-vf", f"setpts=PTS/{sp}", "-af", f"atempo={sp}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", path)
            out.append(path)
        os.remove(seg)
    return out


def count_sheet(track: P.PoseTrack, angles: dict, beats, out_pdf: str, mirror: bool, title: str = "") -> int:
    """One page per 8-count: stick pose on each count with a cue. Returns the number of pages.

    Pass an already flipped track (compare.flip_track) when mirror is True; mirror only
    changes the wording. Left/right always mean the dancer's own sides; in the flipped
    picture those match your sides when you face the screen.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    usable = usable_mask(track)
    jok = joint_ok(track)
    wh = np.array([track.width, track.height])
    segs = eight_counts(beats)
    side = "the dancer's = your" if mirror else "the dancer's"
    with PdfPages(out_pdf) as pdf:
        for n, (b0, _) in enumerate(segs, start=1):
            fig, axes = plt.subplots(2, 4, figsize=(11, 8.5))
            fig.suptitle(f"{title}  8-count {n}  ({beats.times[b0]:.1f}s)\n"
                         f"Left/right = {side} side" + (" (picture mirrored: face the screen and use the arm on the same side)" if mirror else ""),
                         fontsize=11)
            prev = None
            for c in range(8):
                ax = axes[c // 4, c % 4]
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
                for s in ax.spines.values():
                    s.set_color("0.85")
                bi = b0 + c
                if bi >= len(beats.times):
                    ax.set_title(f"{c + 1}", fontsize=16, loc="left")
                    continue
                i = min(int(round(beats.times[bi] * track.fps)), len(track.image) - 1)
                cur = {k: angles[k][i] for k in CUE_JOINTS} | {"torso_twist": angles["torso_twist"][i]}
                if usable[i]:
                    pts = track.image[i, :, :2] * wh
                    hip = (pts[P.L_HIP] + pts[P.R_HIP]) / 2
                    torso = np.linalg.norm((pts[P.L_SHOULDER] + pts[P.R_SHOULDER]) / 2 - hip)
                    q = (pts - hip) / max(torso, 1e-6)
                    for a, b in P.EDGES:
                        if not (jok[i, a] and jok[i, b]):
                            continue
                        col = "#1f77b4" if a in P.LEFT_SIDE and b in P.LEFT_SIDE else (
                            "#ff7f0e" if a not in P.LEFT_SIDE and b not in P.LEFT_SIDE else "0.5")
                        ax.plot([q[a, 0], q[b, 0]], [-q[a, 1], -q[b, 1]], color=col, lw=3, solid_capstyle="round")
                    head = q[P.NOSE]
                    if jok[i, P.NOSE]:
                        ax.add_patch(plt.Circle((head[0], -head[1]), 0.18, fill=False, color="0.3", lw=2))
                    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-3.2, 2.0)
                    cue = "start" if prev is None else cue_text(prev, cur)
                else:
                    ax.text(0.5, 0.5, "not visible\nin footage", ha="center", va="center", color="0.5",
                            transform=ax.transAxes)
                    cue = ""
                ax.set_title(f"{c + 1}", fontsize=16, loc="left", fontweight="bold")
                ax.set_xlabel(cue, fontsize=9, wrap=True)
                prev = cur
            fig.text(0.5, 0.01, "blue = left side, orange = right side. Cues compare each count "
                     "with the previous one; changes under 25 degrees are ignored.", ha="center", fontsize=8,
                     color="0.4")
            fig.tight_layout(rect=(0, 0.03, 1, 0.94))
            pdf.savefig(fig)
            fig.savefig(out_pdf.replace(".pdf", f"_{n:02d}.png"), dpi=80)
            plt.close(fig)
    return len(segs)
