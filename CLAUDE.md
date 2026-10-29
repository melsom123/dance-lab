# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`dancelab` is a Python CLI for studying dance videos: it extracts body pose per frame, turns it into joint angles, lines movement up against the music's beat, and compares a user's recording against a reference dancer. The owner's main machine is Windows; the code must stay cross-platform (paths via `os.path`, ffmpeg called as a subprocess).

## Commands

```
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt                       # ffmpeg must also be on PATH

python -m dancelab analyze VIDEO [--start S --end E] [-o OUTDIR] [--bpm GUESS]
python -m dancelab compare REF_VIDEO YOUR_VIDEO [--ref-start S --ref-end E] [--mirror] [-o OUTDIR]

yt-dlp -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]" -o ref.mp4 URL   # fetch a reference clip
```

There is no test suite or linter yet. The smoke tests used so far:
- `compare` a video against itself: every joint's mean difference must print 0.0 (checks DTW alignment and indexing).
- `compare` a video against a slowed (`setpts=1.25*PTS`), re-encoded copy: differences should stay small (measured 2–12°). Add `hflip` and `--mirror` to exercise the mirror path.

On headless Linux, MediaPipe needs `libegl1` and `libgles2` installed.

## Architecture

Data flows through one object, `pose.PoseTrack` (fps, frame size, and per-frame arrays `image` (T,33,3) normalized image coords, `world` (T,33,3) metric hip-centred 3D coords with y pointing **down**, and `vis` (T,33)). Frames with no detected person are NaN rows; `fill_gaps` interpolates dropouts up to 0.5 s and leaves longer ones NaN, so downstream code must tolerate NaN.

- `pose.py`: MediaPipe Pose Landmarker (tasks API, VIDEO mode, single person). Landmark index constants and the skeleton `EDGES` live here. The heavy model downloads to `models/` on first use (override with `DANCELAB_POSE_MODEL`).
- `pose.py` detects up to `MAX_PEOPLE`, follows the same person frame to frame (nearest hip position; the largest person after a cut), and records how many similar-sized people are in frame (`PoseTrack.people`).
- `features.py`: `joint_ok` marks each landmark visible and in frame; `usable_mask` keeps frames with a single dancer whose torso (shoulders + hips) is visible, for runs of 1 s or more, which rejects tight close-ups. With `--solo` (`PoseTrack.solo_only`) it also rejects frames with several people, which removes film audience shots but would wipe out group dances or bystanders, so it is off by default. Each angle is additionally NaN'd when any of its own landmarks is not `joint_ok`, so half-body shots still give arm/torso data. MediaPipe hallucinates off-screen limbs, which is why this masking matters. Joint angles are computed from **world** coords (3D, less camera-angle dependent), after Savitzky-Golay smoothing. Torso twist uses the floor plane (x, z). `hip_sway`/`hip_bounce` use **image** coords relative to a 2 s moving average, divided by the per-frame torso length in pixels, so zoom changes and travelling don't dominate. `motion_energy` (summed joint speed) is the proxy for movement accents.
- `music.py`: librosa beat tracking on audio extracted by ffmpeg. Count "1" of the 1–8 count is a heuristic (beat phase mod 4 with the strongest onsets). `hit_offsets` matches each beat to the nearest motion-energy peak within half a beat (negative = ahead of the beat).
- `render.py`: OpenCV overlay video (skeleton, beat flash, count, angle readouts, energy strip), written silent then muxed with the source audio via ffmpeg; matplotlib angle plot.
- `compare.py`: DTW (`librosa.sequence.dtw`) on a matrix of `COMPARE_JOINTS` angles to map each reference frame to a user frame, then per-joint absolute error. The comparison video renders on the reference timeline with the reference skeleton drawn as a green ghost, re-anchored to the user's hip centre and scaled by torso length. `--mirror` swaps left/right landmarks via `MIRROR_IDX` and negates x before computing angles; the ghost is mirrored separately at render time.
- `fill.py` (`analyze --fill`): builds a continuous full-body estimate in a hip-centred, torso-length-normalized space and splices it with measured frames, recording a per-joint source (measured / model guess for off-screen limbs / interpolated for gaps up to 1.5 s / borrowed from the same 8-count position whole 8-counts away for longer gaps / none before the first and after the last usable frame). `render.filled_stick_video` draws measured limbs in colour and estimates in grey with a footage inset. Known blind spot: MediaPipe sometimes reports a confident, self-consistent whole body in a legs-only close-up, which passes as measured.
- `cli.py`: argument parsing, `prepare` (ffmpeg trim, and an automatic H.264 re-encode when OpenCV can't decode the source, e.g. YouTube AV1), and output files (`angles.csv`, `beats.csv`, `angles.png`, `overlay.mp4`, `stick.mp4` (skeleton only, centred and scaled to a steady size), `report.txt`, and with `--fill` `stick_filled.mp4` + `angles_filled.csv` / `compare.mp4`, `compare_report.txt`). Pose extraction is cached as `poses.npz` / `ref_poses.npz` / `you_poses.npz` in the output dir and reused when newer than the video, so delete the cache after changing `pose.py`.

## Interpreting results

Single-camera 3D pose is noisy: expect roughly 5–15° error on elbows and shoulders, less on knees and hips, and worse when the dancer faces away, is small in frame, or limbs overlap. Don't present angle differences under about 10–15° as real technique differences. Left/right refer to the dancer's own body (blue = left, orange = right in overlays).
