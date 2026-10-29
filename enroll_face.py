"""
=============================================================================
  Laboratory Access Control System — Pi Camera Enrollment Utility
  (With Anti-Spoofing & Live Eye-Blink Verification)
=============================================================================
  Usage:
    python enroll_face.py "Faculty Name"
    or
    python enroll_face.py --id <profile_id_uuid>
=============================================================================
"""

import os
import sys
import time
import logging
import numpy as np
import cv2
import face_recognition
from supabase import create_client

from door_guard import (
    _l2_norm,
    _check_2d_spoof_artifacts,
    _extract_face_geometry,
    LivenessTracker,
    ENROLLMENT_SAMPLES_REQUIRED,
    ENROLLMENT_TIMEOUT_SECONDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("enroll_face")

# Load .env if present
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://nbcoxrxywbrcjldccinr.supabase.co")
SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5iY294cnh5d2JyY2psZGNjaW5yIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODM2MDc1MTIsImV4cCI6MjA5OTE4MzUxMn0.BbPRDVKfCq0ov-EF2DW90E5OnwnGM4wBnTC-QLtzrSo",
)


def enroll_user(target: str) -> bool:
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1. Resolve Profile in Supabase
    log.info("Searching for profile '%s' in Supabase...", target)
    if "-" in target and len(target) > 30:  # assume UUID
        res = client.table("profiles").select("*").eq("id", target).execute()
    else:
        res = client.table("profiles").select("*").ilike("name", f"%{target}%").execute()

    if not res.data:
        log.error("No profile found matching '%s'. Create profile in App first.", target)
        return False

    profile = res.data[0]
    profile_id = profile["id"]
    profile_name = profile.get("name", "Unknown")
    log.info("Target Profile: %s (ID: %s)", profile_name, profile_id)

    # 2. Open Camera Stream & Perform Live-Verified Multi-Frame Capture
    log.info("Initializing camera... Stand in front of camera and BLINK naturally.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Unable to open camera (/dev/video0).")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    tracker = LivenessTracker()
    collected_embeddings: list[np.ndarray] = []
    last_emb_time = 0.0
    start_t = time.time()

    log.info(
        "🛡️ Live verification active (%.0fs window): Please look at the camera and BLINK...",
        ENROLLMENT_TIMEOUT_SECONDS,
    )

    try:
        while (time.time() - start_t) < ENROLLMENT_TIMEOUT_SECONDS:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                time.sleep(0.04)
                continue

            now = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            small_locs = face_recognition.face_locations(small_rgb)

            if not small_locs:
                tracker.update(None, now=now)
                time.sleep(0.04)
                continue

            best_small = max(small_locs, key=lambda b: max(0, b[2] - b[0]) * max(0, b[1] - b[3]))
            h, w = frame.shape[:2]
            full_box = (
                max(0, min(h - 1, best_small[0] * 2)),
                max(1, min(w, best_small[1] * 2)),
                max(1, min(h, best_small[2] * 2)),
                max(0, min(w - 1, best_small[3] * 2)),
            )

            is_suspect_2d, spoof_reason, lap_var = _check_2d_spoof_artifacts(frame, full_box)
            lm_list = face_recognition.face_landmarks(rgb, [full_box])
            geom = _extract_face_geometry(lm_list[0], full_box, lap_var) if lm_list else None
            is_live, _ = tracker.update(geom, is_suspect_2d, spoof_reason, now=now)

            if (
                geom is not None
                and not is_suspect_2d
                and geom["ear"] >= 0.18
                and (now - last_emb_time) >= 0.30
                and len(collected_embeddings) < (ENROLLMENT_SAMPLES_REQUIRED + 2)
            ):
                if len(collected_embeddings) < (ENROLLMENT_SAMPLES_REQUIRED - 1) or is_live:
                    encs = face_recognition.face_encodings(rgb, [full_box])
                    if encs:
                        cand_emb = _l2_norm(np.array(encs[0], dtype=np.float64))
                        if collected_embeddings:
                            if float(1.0 - np.dot(cand_emb, collected_embeddings[0])) > 0.048:
                                log.warning("Face changed during enrollment — resetting samples.")
                                collected_embeddings.clear()
                                tracker.reset()
                                continue
                        collected_embeddings.append(cand_emb)
                        last_emb_time = now

            if is_live and len(collected_embeddings) >= ENROLLMENT_SAMPLES_REQUIRED:
                embedding = _l2_norm(np.mean(np.stack(collected_embeddings, axis=0), axis=0))
                embedding_list = [float(val) for val in embedding]
                log.info("Updating Supabase profile with live-verified 128-d face embedding...")
                client.table("profiles").update({
                    "face_embedding": embedding_list,
                    "camera_status": "completed",
                }).eq("id", profile_id).execute()
                log.info("✅ SUCCESS: Live-verified face embedding registered for '%s'!", profile_name)
                return True

            time.sleep(0.03)
    finally:
        cap.release()

    log.error(
        "🚫 ENROLLMENT BLOCKED: Liveness verification failed (static photo/calendar or no eye blink detected)."
    )
    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python enroll_face.py <Profile Name or Profile ID>")
        sys.exit(1)

    arg = sys.argv[2] if (sys.argv[1] == "--id" and len(sys.argv) >= 3) else sys.argv[1]
    success = enroll_user(arg)
    sys.exit(0 if success else 1)
