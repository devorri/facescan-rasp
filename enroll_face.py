"""
=============================================================================
  Laboratory Access Control System — Pi Camera Enrollment Utility
  (Random head-turn challenge + 3D parallax liveness, same as door_guard)
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
import cv2
from supabase import create_client

import threading

import door_guard as dg
from door_guard import reload_known_faces, capture_verified_live_enrollment

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
        log.info("No existing profile found for '%s' — creating new Faculty profile in Supabase...", target)
        created = client.table("profiles").insert({
            "name": target.strip(),
            "role": "Faculty - CCS",
            "camera_status": "idle",
        }).execute()
        if not created.data:
            log.error("Failed to create profile '%s' in Supabase.", target)
            return False
        profile = created.data[0]
    else:
        profile = res.data[0]

    profile_id = profile["id"]
    profile_name = profile.get("name", "Unknown")
    log.info("Target Profile: %s (ID: %s)", profile_name, profile_id)

    # 2. Open the camera and run the SAME live-challenge enrollment used by door_guard
    #    (stop the door_guard service first: `sudo systemctl stop door_guard`,
    #     only one program can hold the camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Unable to open camera (/dev/video0). Is door_guard.py still running?")
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    latest = {"frame": None}
    stop = threading.Event()

    def _reader() -> None:
        while not stop.is_set():
            ret, frame = cap.read()
            if ret and frame is not None:
                latest["frame"] = frame
            else:
                time.sleep(0.02)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    # Needed for the "already registered as someone else" duplicate check
    dg.LABORATORY_ID = os.environ.get("LABORATORY_ID", "") or dg.LABORATORY_ID
    reload_known_faces(client)

    def _show(text: str) -> None:
        log.info("👉 %s", text)

    try:
        embedding, message, status_code = capture_verified_live_enrollment(
            user_name=profile_name,
            profile_id=profile_id,
            on_progress=_show,
            frame_source=lambda: latest["frame"],
        )
    finally:
        stop.set()
        reader.join(timeout=1.0)
        cap.release()

    if embedding is None:
        client.table("profiles").update({"camera_status": status_code}).eq("id", profile_id).execute()
        log.error("🚫 ENROLLMENT FAILED: %s", message)
        return False

    client.table("profiles").update({
        "face_embedding": [float(v) for v in embedding],
        "camera_status": "completed",
    }).eq("id", profile_id).execute()
    log.info("✅ SUCCESS: Face embedding registered for '%s'!", profile_name)
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python enroll_face.py <Profile Name or Profile ID>")
        sys.exit(1)

    arg = sys.argv[2] if (sys.argv[1] == "--id" and len(sys.argv) >= 3) else sys.argv[1]
    success = enroll_user(arg)
    sys.exit(0 if success else 1)
