"""dancelab command line.

  python -m dancelab analyze VIDEO [-o OUTDIR] [--start S --end E]
  python -m dancelab compare REF_VIDEO YOUR_VIDEO [-o OUTDIR] [--mirror]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess

import numpy as np

from . import compare as C
from . import fill as F
from . import music, render
from .features import joint_angles, joint_ok, motion_energy, usable_mask, usable_segments
from .pose import DEFAULT_MODEL, PoseTrack, extract


def _opencv_can_read(video):
    import cv2
    cap = cv2.VideoCapture(video)
    ok, _ = cap.read()
    cap.release()
    return ok


def prepare(video, start, end, outdir, name="clip"):
    """Trim if asked, and re-encode to H.264 when OpenCV can't decode the source (e.g. YouTube AV1)."""
    if start is None and end is None and _opencv_can_read(video):
        return video
    out = os.path.join(outdir, f"{name}.mp4")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(video):
        return out
    print(f"Re-encoding {video} -> {out}")
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video]
    if end is not None:
        cmd += ["-t", str(end - (start or 0))]
    subprocess.run(cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", out], check=True)
    return out


def load_or_extract(video, outdir, model, name="poses"):
    cache = os.path.join(outdir, f"{name}.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(video):
        return PoseTrack.load(cache)
    print(f"Extracting poses from {video} ...")
    t = extract(video, model)
    t.save(cache)
    return t


def cmd_analyze(a):
    os.makedirs(a.outdir, exist_ok=True)
    video = prepare(a.video, a.start, a.end, a.outdir)
    track = load_or_extract(video, a.outdir, a.model)
    track.solo_only = a.solo
    detected = np.isfinite(track.image[:, 0, 0]).mean()
    usable = usable_mask(track)
    jok = joint_ok(track)
    angles = joint_angles(track)
    energy = motion_energy(track)
    beats = music.track_beats(video, a.bpm)
    hits = music.hit_offsets(energy, track.fps, beats) if beats else None
    if hits is not None and len(hits):
        bf = np.clip((beats.times * track.fps).round().astype(int), 0, len(usable) - 1)
        hits[~usable[bf]] = np.nan

    with open(os.path.join(a.outdir, "angles.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        names = list(angles)
        wr.writerow(["time_s", "beat_count"] + names + ["movement_energy"])
        for i, t in enumerate(track.times):
            c = ""
            if beats is not None and len(beats.times):
                k = np.searchsorted(beats.times, t, side="right") - 1
                c = int(beats.counts[k]) if k >= 0 else ""
            wr.writerow([f"{t:.3f}", c] + [f"{angles[n][i]:.1f}" for n in names] + [f"{energy[i]:.3f}"])
    if beats is not None:
        with open(os.path.join(a.outdir, "beats.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["beat", "time_s", "count", "accent_offset_ms"] + list(angles))
            for k, bt in enumerate(beats.times):
                i = min(int(round(bt * track.fps)), len(track.image) - 1)
                off = "" if hits is None or not np.isfinite(hits[k]) else f"{hits[k] * 1000:.0f}"
                wr.writerow([k, f"{bt:.3f}", beats.counts[k], off] + [f"{angles[n][i]:.1f}" for n in angles])

    render.plot_angles(angles, energy, track.fps, beats, os.path.join(a.outdir, "angles.png"), hits,
                       title=os.path.basename(a.video))
    for view, stick in (("overlay", False), ("stick", True)):
        if a.view in (view, "both"):
            print(f"Rendering {view} video ...")
            render.overlay_video(video, track, angles, energy, beats, os.path.join(a.outdir, f"{view}.mp4"),
                                 usable, jok, stick=stick)

    if a.fill:
        fp = F.fill(track, beats)
        print("Rendering filled stick video ...")
        render.filled_stick_video(video, track, fp, beats, os.path.join(a.outdir, "stick_filled.mp4"))
        fa = joint_angles(F.filled_track(track, fp))
        with open(os.path.join(a.outdir, "angles_filled.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            names = [n for n in fa if not n.startswith("hip_")]
            wr.writerow(["time_s", "source"] + names)
            for i, t in enumerate(track.times):
                vals = [f"{fa[n][i]:.1f}" if fp.estimated[i] else "" for n in names]
                wr.writerow([f"{t:.3f}", F.SOURCE_NAMES[fp.frame_source(i)]] + vals)
        share = {F.SOURCE_NAMES[k]: (np.array([fp.frame_source(i) for i in range(len(track.times))]) == k).mean()
                 for k in F.SOURCE_NAMES}

    lines = [f"Video: {a.video}", f"Frames: {len(track.image)} @ {track.fps:.1f} fps, person detected in {detected:.0%}, "
             f"dancer usable in {usable.mean():.0%} "
             f"(legs in view in {(usable & jok[:, [25, 26, 27, 28]].all(axis=1)).mean():.0%})"]
    segs = [(s, e) for s, e in usable_segments(usable, track.fps) if e - s >= 3]
    if segs:
        lines.append("Usable sections (3 s or longer): " + ", ".join(f"{s:.0f}-{e:.0f}s" for s, e in segs))
    if beats is not None:
        lines.append(f"Tempo: {beats.tempo:.1f} BPM, {len(beats.times)} beats")
        if hits is not None and np.isfinite(hits).any():
            h = hits[np.isfinite(hits)] * 1000
            lines.append(f"Movement accents vs beat: median {np.median(h):+.0f} ms "
                         f"(negative = ahead of the beat), spread (IQR) {np.subtract(*np.percentile(h, [75, 25])):.0f} ms, "
                         f"{len(h)}/{len(hits)} beats have a clear accent")
    else:
        lines.append("No audio track found; beat analysis skipped.")
    if a.fill:
        lines.append("Filled version (stick_filled.mp4): " + ", ".join(f"{k} {v:.0%}" for k, v in share.items()))
    rng = {n: (np.nanpercentile(v, 5), np.nanpercentile(v, 95)) for n, v in angles.items()}
    lines.append("Range of motion (5th-95th percentile):")
    lines += [f"  {n:<12} {lo:7.1f} .. {hi:7.1f}" for n, (lo, hi) in rng.items()]
    report = "\n".join(lines)
    open(os.path.join(a.outdir, "report.txt"), "w").write(report + "\n")
    print(report)
    print(f"\nOutputs in {a.outdir}: overlay.mp4 / stick.mp4, angles.png, angles.csv, beats.csv, report.txt")


def cmd_compare(a):
    os.makedirs(a.outdir, exist_ok=True)
    ref_video = prepare(a.ref, a.ref_start, a.ref_end, a.outdir, "ref_clip")
    you_video = prepare(a.you, None, None, a.outdir, "you_clip")
    ref = load_or_extract(ref_video, a.outdir, a.model, "ref_poses")
    you = load_or_extract(you_video, a.outdir, a.model, "you_poses")
    ref.solo_only = you.solo_only = a.solo
    if a.mirror:
        you_for_angles = C.mirror_track(you)
    else:
        you_for_angles = you
    ra, ya = joint_angles(ref), joint_angles(you_for_angles)
    ref_to_you, err = C.align(ref, you_for_angles, ra, ya)
    beats = music.track_beats(ref_video, a.bpm)
    print("Rendering comparison video ...")
    C.comparison_video(ref_video, you_video, ref, you, ref_to_you, err, os.path.join(a.outdir, "compare.mp4"), beats,
                         mirror=a.mirror)
    rep = C.summary(err, ref.fps, beats)
    open(os.path.join(a.outdir, "compare_report.txt"), "w").write(rep + "\n")
    print(rep)


def cmd_practice(a):
    from . import practice as PR
    os.makedirs(a.outdir, exist_ok=True)
    video = prepare(a.video, a.start, a.end, a.outdir)
    track = load_or_extract(video, a.outdir, a.model)
    track.solo_only = a.solo
    beats = music.track_beats(video, a.bpm)
    if beats is None or len(beats.times) < 9:
        raise SystemExit("Need music with at least one full 8-count to build practice material.")
    mirror = not a.no_mirror
    tr = C.flip_track(track) if mirror else track
    angles = joint_angles(tr)
    energy = motion_energy(tr)
    full = os.path.join(a.outdir, "practice_full.mp4")
    print("Rendering practice video ...")
    render.overlay_video(video, tr, angles, energy, beats, full, usable_mask(tr), joint_ok(tr), flip=mirror)
    speeds = [float(x) for x in a.speeds.split(",")]
    clips = PR.practice_clips(full, beats, os.path.join(a.outdir, "clips"), speeds, a.loops)
    pages = PR.count_sheet(tr, angles, beats, os.path.join(a.outdir, "count_sheet.pdf"), mirror,
                           title=os.path.basename(a.video))
    print(f"{len(clips)} practice clips in {a.outdir}/clips, count sheet with {pages} pages: count_sheet.pdf")


def main():
    p = argparse.ArgumentParser(prog="dancelab")
    sub = p.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyze", help="pose + beat analysis of one video")
    an.add_argument("video")
    an.add_argument("-o", "--outdir", default="out")
    an.add_argument("--start", type=float)
    an.add_argument("--end", type=float)
    an.add_argument("--bpm", type=float, default=120, help="starting guess for the tempo tracker")
    an.add_argument("--view", choices=["overlay", "stick", "both"], default="both",
                    help="overlay = skeleton on the footage, stick = skeleton on black (default: both)")
    an.add_argument("--model", default=DEFAULT_MODEL)
    an.add_argument("--solo", action="store_true",
                    help="skip frames with several people in shot (e.g. a film cutting to the audience)")
    an.add_argument("--fill", action="store_true",
                    help="also write stick_filled.mp4 / angles_filled.csv: a continuous full-body estimate "
                         "spliced with the measured segments")
    an.set_defaults(func=cmd_analyze)
    pr = sub.add_parser("practice", help="looped slow-motion clips per 8-count and a printable count sheet")
    pr.add_argument("video")
    pr.add_argument("-o", "--outdir", default="out_practice")
    pr.add_argument("--start", type=float)
    pr.add_argument("--end", type=float)
    pr.add_argument("--speeds", default="0.5,0.75,1", help="comma-separated playback speeds (min 0.5)")
    pr.add_argument("--loops", type=int, default=4)
    pr.add_argument("--no-mirror", action="store_true",
                    help="don't flip the picture (default flips it, like a mirrored tutorial, so you "
                         "copy with the arm on the same side of the screen)")
    pr.add_argument("--solo", action="store_true")
    pr.add_argument("--bpm", type=float, default=120)
    pr.add_argument("--model", default=DEFAULT_MODEL)
    pr.set_defaults(func=cmd_practice)
    cp = sub.add_parser("compare", help="compare your recording against a reference")
    cp.add_argument("ref")
    cp.add_argument("you")
    cp.add_argument("-o", "--outdir", default="out_compare")
    cp.add_argument("--ref-start", type=float)
    cp.add_argument("--ref-end", type=float)
    cp.add_argument("--mirror", action="store_true", help="you danced it mirror-image (facing the screen)")
    cp.add_argument("--bpm", type=float, default=120)
    cp.add_argument("--model", default=DEFAULT_MODEL)
    cp.add_argument("--solo", action="store_true",
                    help="skip frames with several people in shot (e.g. a film cutting to the audience)")
    cp.set_defaults(func=cmd_compare)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
