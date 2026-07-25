"""
=============================================================================
  Laboratory Access Control System — Pi Camera Enrollment Utility
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
import json
import logging
import numpy as np
import cv2
import face_recognition
from supabase import create_client

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
    if "-" in target and len(target) > 30: # assume UUID
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

    # 2. Capture Frame from Pi Camera / Webcam
    log.info("Initializing camera... Stand in front of camera.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Unable to open camera (/dev/video0).")
        return False

    log.info("Capturing in 2 seconds...")
    time.sleep(2)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        log.error("Failed to capture frame from camera.")
        return False

    # 3. Detect and extract face embedding
    log.info("Extracting facial features using dlib ResNet...")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)

    if not encodings:
        log.warning("No face detected in frame. Please check lighting and camera position.")
        return False

    embedding = encodings[0]
    
    # L2-normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    embedding_list = [float(val) for val in embedding]

    # 4. Save embedding to Supabase
    log.info("Updating Supabase profile with 128-d face embedding...")
    client.table("profiles").update({"face_embedding": embedding_list}).eq("id", profile_id).execute()
    log.info("✅ SUCCESS: Face embedding registered for '%s'!", profile_name)
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python enroll_face.py <Profile Name or Profile ID>")
        sys.exit(1)
    
    success = enroll_user(sys.argv[1])
    sys.exit(0 if success else 1)
