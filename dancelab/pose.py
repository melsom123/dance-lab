"""Pose extraction with MediaPipe Pose Landmarker (33 body landmarks per frame)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

# MediaPipe landmark indices we use
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

# Skeleton edges drawn in overlays
EDGES = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (L_ANKLE, L_HEEL), (L_HEEL, L_FOOT), (L_ANKLE, L_FOOT),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (R_ANKLE, R_HEEL), (R_HEEL, R_FOOT), (R_ANKLE, R_FOOT),
]
LEFT_SIDE = {L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_FOOT}

DEFAULT_MODEL = os.environ.get(
    "DANCELAB_POSE_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "models", "pose_landmarker_heavy.task"),
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)


@dataclass
class PoseTrack:
    fps: float
    width: int
    height: int
    image: np.ndarray   # (T, 33, 3) normalized x, y in [0,1] and relative z
    world: np.ndarray   # (T, 33, 3) metres, hip-centred, y points down
    vis: np.ndarray     # (T, 33) visibility in [0,1]; NaN rows where nobody was detected
    people: np.ndarray | None = None  # (T,) number of sizeable people detected in the frame
    solo_only: bool = False  # analysis option (not saved): ignore frames with several people in shot

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.image)) / self.fps

    def save(self, path: str) -> None:
        np.savez_compressed(path, fps=self.fps, width=self.width, height=self.height,
                            image=self.image, world=self.world, vis=self.vis,
                            people=self.people if self.people is not None else np.ones(len(self.image)))

    @classmethod
    def load(cls, path: str) -> "PoseTrack":
        d = np.load(path)
        people = d["people"] if "people" in d else None
        return cls(float(d["fps"]), int(d["width"]), int(d["height"]), d["image"], d["world"], d["vis"], people)


MAX_PEOPLE = 4


def _body_size(landmarks) -> float:
    """Height of the bounding box of the confidently visible landmarks (normalized image units)."""
    ys = [p.y for p in landmarks if p.visibility > 0.5]
    return (max(ys) - min(ys)) if len(ys) >= 2 else 0.0


def _hip_xy(landmarks) -> np.ndarray:
    return np.array([(landmarks[L_HIP].x + landmarks[R_HIP].x) / 2, (landmarks[L_HIP].y + landmarks[R_HIP].y) / 2])


def ensure_model(path: str = DEFAULT_MODEL) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading pose model to {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def extract(video_path: str, model_path: str = DEFAULT_MODEL, max_seconds: float | None = None) -> PoseTrack:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=MAX_PEOPLE,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image, world, vis, people = [], [], [], []
    prev_hip = None
    nan = np.full((33, 3), np.nan)
    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_seconds and i / fps >= max_seconds):
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                              int(round(i * 1000 / fps)))
            if res.pose_landmarks:
                # Several people may be detected (group dances, bystanders, audience shots).
                # Follow the same person as in the previous frame; after a cut or at the start,
                # take the biggest. Also count how many people are of comparable size.
                sizes = np.array([_body_size(lm) for lm in res.pose_landmarks])
                hips = np.array([_hip_xy(lm) for lm in res.pose_landmarks])
                k = int(np.argmax(sizes))
                if prev_hip is not None:
                    big = np.flatnonzero(sizes >= 0.5 * sizes.max())
                    d = np.linalg.norm(hips[big] - prev_hip, axis=1)
                    if d.min() < 0.15:
                        k = int(big[np.argmin(d)])
                prev_hip = hips[k]
                people.append(int((sizes >= 0.5 * sizes[k]).sum()))
                lm, wl = res.pose_landmarks[k], res.pose_world_landmarks[k]
                image.append([[p.x, p.y, p.z] for p in lm])
                world.append([[p.x, p.y, p.z] for p in wl])
                vis.append([p.visibility for p in lm])
            else:
                image.append(nan); world.append(nan); vis.append(np.full(33, np.nan)); people.append(0)
                prev_hip = None
            i += 1
            if i % 300 == 0:
                print(f"  pose: {i} frames ({i / fps:.0f}s)")
    cap.release()
    if not image:
        raise RuntimeError(f"Could not decode any frames from {video_path}")
    track = PoseTrack(fps, w, h, np.array(image), np.array(world), np.array(vis), np.array(people))
    return fill_gaps(track)


def fill_gaps(track: PoseTrack, max_gap_s: float = 0.5) -> PoseTrack:
    """Linearly interpolate short dropouts so angle curves stay continuous."""
    max_gap = int(max_gap_s * track.fps)
    for arr in (track.image, track.world, track.vis):
        flat = arr.reshape(len(arr), -1)
        missing = np.isnan(flat[:, 0])
        if missing.all() or not missing.any():
            continue
        idx = np.arange(len(flat))
        good = ~missing
        # Only fill runs no longer than max_gap
        run_start = None
        for t in range(len(flat) + 1):
            m = t < len(flat) and missing[t]
            if m and run_start is None:
                run_start = t
            elif not m and run_start is not None:
                if t - run_start <= max_gap and run_start > 0 and t < len(flat):
                    for c in range(flat.shape[1]):
                        flat[run_start:t, c] = np.interp(idx[run_start:t], idx[good], flat[good, c])
                run_start = None
    return track
