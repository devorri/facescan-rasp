"""
=============================================================================
  liveness.py — Active challenge-response liveness + 3D parallax check
=============================================================================
  Why this exists
  ---------------
  The old tracker decided "live vs spoof" from raw head-yaw jitter:
    * a printed photo tilted/rotated in the hand changed the (non roll-
      compensated) yaw, so photos could PASS;
    * a real person standing still for 2.5 s was labelled
      "SPOOF BLOCKED: Static Photo", so real people FAILED even after moving
      (the window, the 8 s expiry and the reset-after-decision all re-armed it).

  How this works
  --------------
  1. Pose is measured roll-compensated (eye line rotated to horizontal):
        yaw   = nose-tip x offset from eye-midpoint / inter-ocular distance
        pitch = nose-tip y / mouth y  (both measured from the eye line)
     Tilting or rotating a flat photo does NOT change these ratios, a real
     head turn does.
  2. The user gets a RANDOM sequence of prompts (LEFT / RIGHT / UP / DOWN) so a
     pre-recorded video cannot know what to do.
  3. At each turn we fit a homography between the "looking straight" landmarks
     and the turned landmarks using eyebrows/eyes/mouth corners, then measure
     how far the NOSE deviates from that plane. A flat photo/screen moves as one
     plane (deviation == landmark noise); a real 3D nose sticks out (parallax).
  4. A face track must stay continuous (no jumps / swaps) during the challenge.

  Nothing in face anti-spoofing is 100 % — a high-quality replayed video of the
  same person doing the same random moves, or a 3D mask, can still beat it.
  This combination blocks printed photos, ID cards, calendars and phone/laptop
  screens showing a still photo, which is what the client was testing with.
=============================================================================
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tunables (all overridable from .env)
# ---------------------------------------------------------------------------
YAW_TURN_THRESHOLD: float = float(os.environ.get("LIVENESS_YAW_THRESHOLD", "0.085"))      # ≈12° head turn
PITCH_TURN_THRESHOLD: float = float(os.environ.get("LIVENESS_PITCH_THRESHOLD", "0.075"))  # ≈11° nod
CENTER_TOLERANCE: float = float(os.environ.get("LIVENESS_CENTER_TOLERANCE", "0.05"))
PARALLAX_MIN: float = float(os.environ.get("LIVENESS_PARALLAX_MIN", "0.045"))
PARALLAX_NOISE_FACTOR: float = float(os.environ.get("LIVENESS_PARALLAX_NOISE_FACTOR", "1.6"))
HOLD_FRAMES: int = int(os.environ.get("LIVENESS_HOLD_FRAMES", "2"))
# Camera delivers a mirrored image (some USB webcams do). Flips LEFT/RIGHT.
CAMERA_MIRRORED: bool = os.environ.get("CAMERA_MIRRORED", "0") == "1"

LEFT, RIGHT, UP, DOWN = "LEFT", "RIGHT", "UP", "DOWN"

PROMPTS = {
    LEFT: "Turn head to YOUR LEFT  <<",
    RIGHT: "Turn head to YOUR RIGHT  >>",
    UP: "Look UP  ^",
    DOWN: "Look DOWN  v",
    "CENTER": "Look straight at the camera",
}

# Indices into the 50-point array built by landmarks_to_array()
_ANCHORS = list(range(17, 27)) + list(range(36, 48)) + [48, 49]   # brows, eyes, mouth corners
_NOSE = list(range(28, 36))                                        # nose bridge (lower) + nose tip
_NOSE_TIP = 33


def landmarks_to_array(lm: dict) -> np.ndarray | None:
    """face_recognition landmark dict -> (50, 2) float array in dlib 68-pt order.

    0-16 jaw, 17-26 brows, 27-30 nose bridge, 31-35 nose tip, 36-47 eyes,
    48 = left mouth corner, 49 = right mouth corner.
    """
    try:
        pts: list = []
        for key, n in (
            ("chin", 17), ("left_eyebrow", 5), ("right_eyebrow", 5),
            ("nose_bridge", 4), ("nose_tip", 5), ("left_eye", 6), ("right_eye", 6),
        ):
            seg = lm.get(key) or []
            if len(seg) != n:
                return None
            pts.extend(seg)
        top_lip = lm.get("top_lip") or []
        if len(top_lip) < 7:
            return None
        pts.append(top_lip[0])
        pts.append(top_lip[6])
        return np.asarray(pts, dtype=np.float64)
    except Exception:
        return None


def inter_ocular(p: np.ndarray) -> float:
    return float(np.linalg.norm(p[42:48].mean(0) - p[36:42].mean(0)))


def head_pose(p: np.ndarray) -> tuple[float, float, float]:
    """Roll-compensated (yaw, pitch, inter_ocular_px).

    yaw > 0  -> nose moved to image-right (person turned to THEIR left on a
                non-mirrored camera)
    pitch    -> nose-tip height / mouth height below the eye line
                (smaller = looking UP, bigger = looking DOWN)
    """
    le = p[36:42].mean(0)
    re = p[42:48].mean(0)
    axis = re - le
    io = float(np.linalg.norm(axis))
    if io < 1.0:
        return 0.0, 0.0, 0.0
    ang = np.arctan2(axis[1], axis[0])
    c, s = np.cos(-ang), np.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    q = (p - 0.5 * (le + re)) @ rot.T
    mouth_y = 0.5 * (q[48, 1] + q[49, 1])
    yaw = float(q[_NOSE_TIP, 0] / io)
    pitch = float(q[_NOSE_TIP, 1] / mouth_y) if mouth_y > 1.0 else 0.0
    return yaw, pitch, io


def planar_parallax(a: np.ndarray, b: np.ndarray) -> float:
    """How far the nose departs from a single-plane (photo/screen) motion a->b.

    Returns mean nose deviation / inter-ocular distance. ~0.02-0.035 for a flat
    photo (landmark noise), >= 0.05 for a real head turned ~12°+.
    """
    try:
        H, _ = cv2.findHomography(
            a[_ANCHORS].astype(np.float32), b[_ANCHORS].astype(np.float32), 0
        )
        if H is None:
            return 0.0
        proj = cv2.perspectiveTransform(
            a[_NOSE].reshape(-1, 1, 2).astype(np.float32), H
        ).reshape(-1, 2)
        io = max(1.0, inter_ocular(b))
        return float(np.mean(np.linalg.norm(proj - b[_NOSE], axis=1)) / io)
    except Exception:
        return 0.0


def _direction_satisfied(direction: str, d_yaw: float, d_pitch: float) -> bool:
    sign = -1.0 if CAMERA_MIRRORED else 1.0
    if direction == LEFT:
        return sign * d_yaw >= YAW_TURN_THRESHOLD
    if direction == RIGHT:
        return sign * d_yaw <= -YAW_TURN_THRESHOLD
    if direction == UP:
        return d_pitch <= -PITCH_TURN_THRESHOLD
    if direction == DOWN:
        return d_pitch >= PITCH_TURN_THRESHOLD
    return False


def _wrong_way(direction: str, d_yaw: float, d_pitch: float) -> bool:
    opposite = {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}[direction]
    return _direction_satisfied(opposite, d_yaw, d_pitch)


@dataclass
class ChallengeResult:
    state: str              # "calibrating" | "challenge" | "recenter" | "passed" | "failed"
    prompt: str             # text to show the user
    is_live: bool = False
    is_spoof: bool = False  # True only when we have actual evidence of a flat face
    progress: tuple[int, int] = (0, 0)
    at_center: bool = False
    detail: str = ""


@dataclass
class ChallengeSession:
    """Random head-movement challenge with 3D parallax verification.

    Feed one landmark array per processed frame via update(). The session is
    bound to one continuous face track; call reset() when the face is lost.
    """

    num_challenges: int = 1
    directions: tuple[str, ...] = (LEFT, RIGHT)
    require_recenter: bool = False
    timeout_s: float = 12.0
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        pool = list(self.directions)
        self.rng.shuffle(pool)
        while len(pool) < self.num_challenges:  # allow repeats if few directions
            pool.append(self.rng.choice(self.directions))
        self.sequence: list[str] = pool[: self.num_challenges]
        self.idx = 0
        self.state = "calibrating"
        self.started_at = 0.0
        self.baseline_frames: list[np.ndarray] = []
        self.baseline: np.ndarray | None = None
        self.base_yaw = 0.0
        self.base_pitch = 0.0
        self.noise_floor = 0.0
        self.recent: list[np.ndarray] = []
        self.hold = 0
        self.parallax_scores: list[float] = []
        self.flat_hits = 0
        self.last_center: np.ndarray | None = None
        self.passed_at = 0.0
        self.fail_reason = ""

    @property
    def current_direction(self) -> str | None:
        return self.sequence[self.idx] if self.idx < len(self.sequence) else None

    def offset(self, pts: np.ndarray) -> tuple[float, float] | None:
        """(d_yaw, d_pitch) of pts relative to the calibrated straight-ahead pose."""
        if self.baseline is None:
            return None
        yaw, pitch, _ = head_pose(pts)
        return yaw - self.base_yaw, pitch - self.base_pitch

    def is_centered(self, pts: np.ndarray) -> bool:
        off = self.offset(pts)
        return off is not None and abs(off[0]) < CENTER_TOLERANCE and abs(off[1]) < CENTER_TOLERANCE

    def _median(self, frames: list[np.ndarray]) -> np.ndarray:
        return np.median(np.stack(frames, axis=0), axis=0)

    # ----------------------------------------------------------------- update
    def update(self, pts: np.ndarray | None, now: float | None = None) -> ChallengeResult:
        now = time.time() if now is None else now
        total = len(self.sequence)

        if self.state == "passed":
            return ChallengeResult("passed", "Live face verified", True, progress=(total, total), at_center=True)
        if self.state == "failed":
            return ChallengeResult("failed", self.fail_reason, False, is_spoof=True,
                                   progress=(self.idx, total), detail=self.fail_reason)

        if pts is None:
            return ChallengeResult(self.state, PROMPTS["CENTER"], progress=(self.idx, total))

        if self.started_at == 0.0:
            self.started_at = now

        # Track continuity — a sudden jump means a different face/photo was swapped in
        if self.last_center is not None:
            io = max(1.0, inter_ocular(pts))
            jump = float(np.linalg.norm(pts.mean(0) - self.last_center))
            if jump > 0.8 * io:
                self.reset()
                self.started_at = now
        self.last_center = pts.mean(0)

        if (now - self.started_at) > self.timeout_s:
            # Timed out: not evidence of spoofing unless we saw flat motion
            if self.flat_hits >= 2:
                self.state = "failed"
                self.fail_reason = "Flat photo/screen motion detected (no 3D depth)"
                return ChallengeResult("failed", self.fail_reason, is_spoof=True,
                                       progress=(self.idx, total), detail=self.fail_reason)
            self.reset()
            self.started_at = now
            return ChallengeResult("calibrating", "Let's try again — " + PROMPTS["CENTER"], progress=(0, total))

        # ---------------- calibrate a "looking straight" baseline -----------------
        if self.state == "calibrating":
            self.baseline_frames.append(pts)
            self.baseline_frames = self.baseline_frames[-5:]
            if len(self.baseline_frames) >= 4:
                yaws = [head_pose(f)[0] for f in self.baseline_frames]
                pitches = [head_pose(f)[1] for f in self.baseline_frames]
                stable = (max(yaws) - min(yaws)) < 0.06 and (max(pitches) - min(pitches)) < 0.06
                if stable:
                    self.baseline = self._median(self.baseline_frames)
                    self.base_yaw, self.base_pitch, _ = head_pose(self.baseline)
                    half = len(self.baseline_frames) // 2
                    self.noise_floor = planar_parallax(
                        self._median(self.baseline_frames[:half]),
                        self._median(self.baseline_frames[half:]),
                    )
                    self.state = "challenge"
                    self.recent = []
                    self.hold = 0
                    return ChallengeResult("challenge", PROMPTS[self.current_direction],
                                           progress=(self.idx, total), at_center=True)
            return ChallengeResult("calibrating", PROMPTS["CENTER"], progress=(self.idx, total))

        # Smooth landmarks over the last 3 frames to beat jitter
        self.recent.append(pts)
        self.recent = self.recent[-3:]
        smooth = self._median(self.recent)
        yaw, pitch, _ = head_pose(smooth)
        d_yaw = yaw - self.base_yaw
        d_pitch = pitch - self.base_pitch
        at_center = abs(d_yaw) < CENTER_TOLERANCE and abs(d_pitch) < CENTER_TOLERANCE

        # ---------------- wait for user to look straight again --------------------
        if self.state == "recenter":
            if at_center:
                self.hold += 1
                if self.hold >= HOLD_FRAMES:
                    self.state = "challenge"
                    self.hold = 0
                    self.recent = []
                    return ChallengeResult("challenge", PROMPTS[self.current_direction],
                                           progress=(self.idx, total), at_center=True)
            else:
                self.hold = 0
            return ChallengeResult("recenter", PROMPTS["CENTER"], progress=(self.idx, total), at_center=at_center)

        # ---------------- active challenge ----------------------------------------
        direction = self.current_direction
        if _direction_satisfied(direction, d_yaw, d_pitch):
            self.hold += 1
            if self.hold >= HOLD_FRAMES:
                par = planar_parallax(self.baseline, smooth)
                needed = max(PARALLAX_MIN, PARALLAX_NOISE_FACTOR * self.noise_floor)
                if par < needed:
                    # Pose ratios changed but the face moved like a flat plane
                    self.flat_hits += 1
                    self.hold = 0
                    self.recent = []
                    if self.flat_hits >= 3:
                        self.state = "failed"
                        self.fail_reason = "Flat photo/screen detected (no 3D depth)"
                        return ChallengeResult("failed", self.fail_reason, is_spoof=True,
                                               progress=(self.idx, total),
                                               detail=f"parallax={par:.3f} need={needed:.3f}")
                    return ChallengeResult("challenge", PROMPTS[direction] + " (a bit more)",
                                           progress=(self.idx, total),
                                           detail=f"parallax={par:.3f} need={needed:.3f}")
                self.parallax_scores.append(par)
                self.idx += 1
                self.hold = 0
                self.recent = []
                if self.idx >= total:
                    self.state = "passed"
                    self.passed_at = now
                    return ChallengeResult("passed", "Live face verified", True,
                                           progress=(total, total),
                                           detail=f"parallax={min(self.parallax_scores):.3f}")
                if self.require_recenter:
                    self.state = "recenter"
                    return ChallengeResult("recenter", PROMPTS["CENTER"], progress=(self.idx, total))
                return ChallengeResult("challenge", PROMPTS[self.current_direction], progress=(self.idx, total))
        else:
            self.hold = 0
            if _wrong_way(direction, d_yaw, d_pitch):
                return ChallengeResult("challenge", "Other way! " + PROMPTS[direction],
                                       progress=(self.idx, total))
        return ChallengeResult("challenge", PROMPTS[direction], progress=(self.idx, total), at_center=at_center)
