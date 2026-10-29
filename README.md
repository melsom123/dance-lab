# dancelab

Study dancers' joint movements against the music, then compare your own recording to theirs.

## Setup (Windows, macOS or Linux; Python 3.10–3.13, plus ffmpeg on PATH)

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
The MediaPipe pose model (~30 MB) downloads on first run into `models/`.

## 1. Get a reference clip

```
yt-dlp -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]" -o ref.mp4 "https://www.youtube.com/watch?v=..."
```
Best results: one dancer, full body in frame (feet included), static camera, facing camera.

## 2. Analyze it

```
python -m dancelab analyze ref.mp4 --start 30 --end 60 -o out_ref
```
Outputs in `out_ref/`:
- `overlay.mp4` skeleton (blue = dancer's left, orange = right), beat flash, 1–8 count, live angles, movement-energy strip with beat ticks
- `angles.png` joint-angle curves with beat lines (red = estimated count 1)
- `angles.csv` per-frame angles; `beats.csv` pose at every beat + how far the nearest movement accent is from it
- `report.txt` tempo, accent timing (ahead/behind the beat), range of motion per joint

## 3. Compare yourself

Record the same section (same framing, full body). Then:
```
python -m dancelab compare ref.mp4 me.mp4 --ref-start 30 --ref-end 60 -o out_cmp [--mirror]
```
Use `--mirror` if you danced it mirror-image (e.g. copying a front-facing video like a mirror).
Your clip is time-aligned to the reference with DTW on joint angles, so you don't need to start at exactly the same moment.
Outputs `compare.mp4` (reference left; you right with the reference pose as a green ghost scaled to your body) and `compare_report.txt` (which joints and which beats differ most).

## Accuracy notes
- Angles are from MediaPipe's 3D "world" landmarks, estimated from a single camera. Expect roughly 5–15° of noise
  on elbows/shoulders, less on knees/hips; worse when the dancer faces away, is small in frame, or limbs overlap.
- Count "1" is a heuristic guess (strongest beat phase); tempo is usually right, downbeat may be off by a beat or two.
- "Movement energy" is total joint speed; its peaks are treated as accents/hits.
