"""
=============================================================================
  Laboratory Access Control System — Raspberry Pi Door Guard
=============================================================================
  Hardware:
    - Raspberry Pi 4
    - Pi Camera Module (or USB webcam on /dev/video0)
    - 12V Relay Module: signal pin (IN) → GPIO 18 (BCM)
    - Solenoid Lock powered by 12V supply with shared GND to Pi

  Dependencies (install before running):
    pip install opencv-python face-recognition supabase RPi.GPIO numpy

  Environment Variables (set in /etc/environment or a .env file):
    SUPABASE_URL  — e.g. https://xyzxyz.supabase.co
    SUPABASE_KEY  — your project's anon/service-role key
    LABORATORY_ID — UUID of the laboratory this Pi guards

  Run:
    python door_guard.py
=============================================================================
"""

import os
import sys
import time
import logging
import threading
import importlib
import json
import numpy as np

# Ensure DISPLAY is set for RealVNC / X11 GUI window
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"

# ---------------------------------------------------------------------------
# Logging — configure before importing project modules so all libs use it
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("door_guard")

# ---------------------------------------------------------------------------
# OpenCV — used for camera capture and frame pre-processing
# ---------------------------------------------------------------------------
import cv2

# ---------------------------------------------------------------------------
# face_recognition — dlib-backed library for face detection and embeddings
# ---------------------------------------------------------------------------
try:
    face_recognition = importlib.import_module("face_recognition")
except ImportError:
    face_recognition = None
    log.warning("face_recognition not installed in current environment.")


# ---------------------------------------------------------------------------
# Supabase Python client
# ---------------------------------------------------------------------------
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Active challenge liveness (random head turns + 3D parallax) — see liveness.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from liveness import (  # noqa: E402
    ChallengeSession,
    landmarks_to_array,
    LEFT as LIVENESS_LEFT,
    RIGHT as LIVENESS_RIGHT,
    UP as LIVENESS_UP,
    DOWN as LIVENESS_DOWN,
)

# ---------------------------------------------------------------------------
# RPi.GPIO — controls the relay that drives the solenoid lock
# ---------------------------------------------------------------------------
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    # Allow the script to be tested on a non-Pi machine without crashing
    log.warning("RPi.GPIO not found — running in SIMULATION mode (no relay).")
    ON_PI = False

# =============================================================================
#  CONFIGURATION & LOCAL ENV PARSER
# =============================================================================

# Helper to load .env file if running manually in terminal
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

# -- Supabase Credentials & Fallbacks ----------------------------------------
SUPABASE_URL: str = os.environ.get(
    "SUPABASE_URL", "https://nbcoxrxywbrcjldccinr.supabase.co"
)
SUPABASE_KEY: str = os.environ.get(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5iY294cnh5d2JyY2psZGNjaW5yIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODM2MDc1MTIsImV4cCI6MjA5OTE4MzUxMn0.BbPRDVKfCq0ov-EF2DW90E5OnwnGM4wBnTC-QLtzrSo",
)

# UUID or Name of the laboratory this Pi is guarding
LABORATORY_NAME: str = os.environ.get("LABORATORY_NAME", "CCS Laboratory")
LABORATORY_ID: str = os.environ.get("LABORATORY_ID", "")

# -- Face Recognition ---------------------------------------------------------
# Stored embeddings are L2-normalised and compared with cosine distance
# (1 - dot(u, v)). dlib's raw 128-d descriptors have a norm of ~1.45, so
#   cos_dist ≈ euclidean² / 4.2      (0.036 ≈ dlib euclidean 0.39)
# Same person under the Pi camera:  ~0.010 – 0.034
# Different (similar-looking) people: ~0.040 – 0.080
# Never raise FACE_MATCH_THRESHOLD above 0.042 or look-alikes will match.
_raw_thresh = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.036"))
FACE_MATCH_THRESHOLD: float        = min(_raw_thresh, 0.042)
FACE_MATCH_MARGIN: float           = float(os.environ.get("FACE_MATCH_MARGIN", "0.012"))  # best must beat 2nd-best profile by this
FACE_MATCH_SURE: float             = 0.015  # below this the match is unambiguous even without margin
FACE_MATCH_VOTES: int              = 3      # live embeddings averaged before deciding who it is
DUPLICATE_FACE_THRESHOLD: float    = float(os.environ.get("DUPLICATE_FACE_THRESHOLD", "0.050"))  # enrollment: reject if another profile is this close
FRAME_SCALE_FACTOR: float          = 0.5    # Downscale factor for fast HOG box detection
MIN_FACE_SIZE_PX: int              = 80     # Minimum face width/height in 640x480 pixels
MIN_ENROLL_FACE_SIZE_PX: int       = 95     # Minimum face size during enrollment

# -- Liveness / Anti-Spoofing --------------------------------------------------
# Active challenge: the person is asked to turn LEFT/RIGHT (and UP/DOWN when
# enrolling) in a random order, and the turn must show real 3D nose parallax.
# See liveness.py for the details.
REQUIRE_LIVENESS: bool             = os.environ.get("REQUIRE_LIVENESS", "1") == "1"
RECOGNITION_CHALLENGES: int        = int(os.environ.get("RECOGNITION_CHALLENGES", "1"))  # 1 random LEFT/RIGHT turn at the door
ENROLLMENT_CHALLENGES: int         = int(os.environ.get("ENROLLMENT_CHALLENGES", "3"))   # random LEFT/RIGHT/UP/DOWN when registering
ENROLLMENT_TIMEOUT_SECONDS: float  = float(os.environ.get("ENROLLMENT_TIMEOUT_SECONDS", "40"))
ENROLLMENT_SAMPLES_REQUIRED: int   = 5      # frontal embeddings averaged into the stored template
# Passive MiniFASNet model (photo/screen texture classifier). Used as an extra
# vote: only blocks when the AVERAGE live probability over several frames is low.
PASSIVE_ANTISPOOF: bool            = os.environ.get("PASSIVE_ANTISPOOF", "1") == "1"
PASSIVE_LIVE_MIN: float            = float(os.environ.get("PASSIVE_LIVE_MIN", "0.25"))
PASSIVE_MIN_FRAMES: int            = 6

# -- Cooldown ----------------------------------------------------------------
COOLDOWN_SECONDS: int = 8           # Minimum seconds between repeated decisions

# -- GPIO / Relay ------------------------------------------------------------
RELAY_PIN: int        = 18          # BCM pin number — Physical Pin 12
RELAY_ACTIVE_LOW: bool = False       # False = LOW signal is OFF (LOCKED), HIGH signal is ON (UNLOCKED)
DOOR_OPEN_SECONDS: int = 5          # How long to keep the solenoid unlocked

# -- Manual Override Button --------------------------------------------------
# Wire: one leg of button → GPIO 27 (Physical Pin 13), other leg → GND
# Internal pull-up is enabled; button press pulls pin LOW.
BUTTON_PIN: int = 27                # BCM pin number — Physical Pin 13
BUTTON_DEBOUNCE_MS: int = 300       # Milliseconds to debounce button presses

# -- Camera ------------------------------------------------------------------
CAMERA_INDEX: int = 0               # 0 = /dev/video0 (Pi Camera or USB cam)

# -- Operating Hours ----------------------------------------------------------
# Access is only granted between these times on the allowed weekdays.
# Weekday numbers: Monday=0, Tuesday=1, ..., Saturday=5  (Sunday=6 is excluded)
OP_HOURS_START: int = 7             # 7:00 AM  (24-h hour)
OP_HOURS_END:   int = 19            # 7:00 PM  (24-h hour, exclusive boundary)
OP_DAYS: set[int] = {0, 1, 2, 3, 4, 5}  # Monday–Saturday
LATE_GRACE_PERIOD_MINUTES: int = 15     # 15 minutes grace period after start_time


# =============================================================================
#  GLOBAL STATE
# =============================================================================

# List of dicts: {"id": str, "name": str, "embedding": np.ndarray}
known_faces: list[dict] = []
known_faces_lock = threading.Lock()  # Protect list from concurrent writes

# List of schedule dicts: [{"id": str, "faculty_id": str, "laboratory_id": str, "day_of_week": str, "start_time": str, "end_time": str}]
known_schedules: list[dict] = []
known_schedules_lock = threading.Lock()

# Timestamp of the last access decision (for cooldown)
last_decision_time: float = 0.0

# Face detection status (used by UI)
face_in_frame_status: bool = False
last_detected_name: str = ""
cam_ready_status: bool = False
latest_frame = None
enrollment_status: str = "idle"
enrollment_name: str = ""
enrollment_capture_active: bool = False

# 🔥 Pi mode control & thread safety lock for dlib / face_recognition
pi_mode: str = "recognition"  # 'recognition' | 'enrollment' | 'idle'
pi_mode_lock = threading.Lock()
dlib_lock = threading.Lock()  # Protect dlib C++ calls from concurrent thread execution
pi_mode_last_sync: float = 0

# 🚀 Asynchronous 30 FPS face recognition state for UI overlay
active_face_locations: list = []
active_face_label: str = ""
active_box_color: tuple[int, int, int] = (0, 255, 0)
face_worker_running: bool = True


def _l2_norm(v: np.ndarray) -> np.ndarray:
    """Return L2‑normalized vector (unit length)."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


# =============================================================================
#  SUPABASE CLIENT
# =============================================================================

def init_supabase() -> Client:
    """Create and return a Supabase client instance."""
    log.info("Connecting to Supabase at %s ...", SUPABASE_URL)
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    log.info("Supabase client initialised.")
    return client


def resolve_laboratory_id(client: Client) -> str:
    """
    Determine the UUID of this laboratory.

    Resolution order:
      1. LABORATORY_ID env var   — use it directly (raw UUID).
      2. LABORATORY_NAME env var — query the `laboratories` table for a match.

    Exits the process if neither resolves to a valid UUID.
    """
    global LABORATORY_ID

    if LABORATORY_ID:
        log.info("Using LABORATORY_ID from environment: %s", LABORATORY_ID)
        return LABORATORY_ID

    if not LABORATORY_NAME:
        log.error(
            "Neither LABORATORY_ID nor LABORATORY_NAME is set. "
            "Set one of these environment variables before starting."
        )
        sys.exit(1)

    log.info("Resolving laboratory UUID for name: '%s' ...", LABORATORY_NAME)
    response = (
        client.table("laboratories")
        .select("id, name")
        .ilike("name", LABORATORY_NAME)
        .limit(1)
        .execute()
    )

    if not response.data:
        log.error(
            "No laboratory found with name '%s'. "
            "Check the name in the `laboratories` table or set LABORATORY_ID directly.",
            LABORATORY_NAME,
        )
        sys.exit(1)

    resolved = response.data[0]["id"]
    log.info("Resolved laboratory '%s' -> id=%s", response.data[0]["name"], resolved)
    LABORATORY_ID = resolved
    return resolved


# =============================================================================
#  PI MODE SYNC WITH DATABASE
# =============================================================================

def sync_pi_mode_with_database(client: Client) -> None:
    """Read the current mode from database and update pi_mode."""
    global pi_mode
    
    if not LABORATORY_ID:
        return
    
    try:
        response = (
            client.table("laboratories")
            .select("mode")
            .eq("id", LABORATORY_ID)
            .single()
            .execute()
        )
        
        if response.data and "mode" in response.data:
            db_mode = response.data["mode"]
            if db_mode in ['recognition', 'enrollment', 'idle']:
                with pi_mode_lock:
                    if pi_mode != db_mode:
                        pi_mode = db_mode
                        log.info(f"🔄 Pi mode synced from database: {pi_mode.upper()}")
    except Exception as e:
        log.debug(f"Mode sync error: {e}")


def push_pi_mode_to_database(client: Client, mode: str) -> bool:
    """Push the current pi_mode to database."""
    global pi_mode
    
    if not LABORATORY_ID:
        return False
    
    try:
        client.table("laboratories").update({
            "mode": mode,
            "pi_status": {
                "mode": mode,
                "last_update": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        }).eq("id", LABORATORY_ID).execute()
        
        log.info(f"✅ Pi mode pushed to database: {mode.upper()}")
        return True
    except Exception as e:
        log.error(f"Failed to push mode to database: {e}")
        return False


def check_and_update_mode(client: Client) -> None:
    """Background thread to periodically sync mode with database."""
    global pi_mode_last_sync
    
    while True:
        time.sleep(5)  # Check every 5 seconds
        
        try:
            # Sync mode from database (app might have changed it)
            sync_pi_mode_with_database(client)
        except Exception as e:
            log.debug(f"Mode sync background error: {e}")


# =============================================================================
#  FACE REGISTRY — load from DB and keep in memory
# =============================================================================

def fetch_known_faces(client: Client) -> list[dict]:
    """
    Query the `profiles` table and return all rows that have a non-null
    face_embedding column.  The vector is stored in Supabase as a JSON array
    and is converted to a numpy float64 array here.
    """
    log.info("Fetching registered face embeddings from Supabase ...")
    response = (
        client.table("profiles")
        .select("id, name, role, face_embedding")
        .not_.is_("face_embedding", "null")
        .execute()
    )

    faces = []
    for row in response.data:
        raw = row.get("face_embedding")
        if raw is None:
            continue
        try:
            if isinstance(raw, str):
                raw = json.loads(raw)
            embedding = np.array(raw, dtype=np.float64)
            # L2‑normalize to make vectors comparable with cosine distance
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            faces.append({
                "id":        row["id"],
                "name":      row.get("name", "Unknown"),
                "role":      row.get("role", "Faculty"),
                "embedding": embedding,
            })
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            log.warning("Skipping profile %s — bad embedding: %s", row["id"], exc)

    log.info("Loaded %d face profile(s) from the database.", len(faces))
    return faces


def reload_known_faces(client: Client) -> None:
    """Thread-safe refresh of the in-memory face registry and schedule registry."""
    fresh_faces = fetch_known_faces(client)
    with known_faces_lock:
        known_faces.clear()
        known_faces.extend(fresh_faces)
    log.info("Face registry refreshed (%d profiles loaded).", len(known_faces))
    reload_schedules(client)


def fetch_schedules(client: Client) -> list[dict]:
    """Fetch all schedule assignments from Supabase for this laboratory."""
    try:
        query = client.table("schedules").select("*")
        if LABORATORY_ID:
            query = query.eq("laboratory_id", LABORATORY_ID)
        response = query.execute()
        schedules = response.data or []
        log.info("Loaded %d schedule entry/entries from database.", len(schedules))
        return schedules
    except Exception as exc:
        log.error("Failed to fetch schedules from database: %s", exc)
        return []


def reload_schedules(client: Client) -> None:
    """Thread-safe refresh of the in-memory schedule registry."""
    fresh_scheds = fetch_schedules(client)
    with known_schedules_lock:
        known_schedules.clear()
        known_schedules.extend(fresh_scheds)


def _normalize_time_str(t_str: str) -> str:
    """Ensure time string is in HH:MM:SS format."""
    parts = str(t_str).strip().split(":")
    if len(parts) == 2:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"
    elif len(parts) >= 3:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
    return t_str.strip()


def check_faculty_schedule_status(faculty_id: str) -> tuple[bool, str, bool, int]:
    """
    Strict schedule validation for faculty with 15-minute grace period.
    Custodians / Administrators have master 24/7 access without requiring a class schedule.
    Returns:
      (is_sched_ok: bool, message: str, is_late: bool, mins_late: int)
    """
    # 1. Check if person is a Custodian / Lab Admin (Master Access)
    with known_faces_lock:
        profile_match = next((p for p in known_faces if str(p.get("id")) == str(faculty_id)), None)
        if profile_match:
            role = str(profile_match.get("role", "")).strip().lower()
            # Only true Custodians or direct Admins get 24/7 master access.
            # Regular faculty (even in Admin departments) must follow scheduled laboratory hours.
            if (role == "custodian" or role == "admin" or role.startswith("custodian")) and not role.startswith("faculty"):
                return True, f"Custodian Master Access ({profile_match.get('role')})", False, 0

    now_struct = time.localtime()
    current_day = time.strftime("%A", now_struct)  # e.g., 'Monday'
    current_time = time.strftime("%H:%M:%S", now_struct)
    current_mins = now_struct.tm_hour * 60 + now_struct.tm_min

    # 2. Hard lab hours for instructors (7:00 AM – 7:00 PM), even if a schedule in
    #    the database says otherwise (e.g. one typed directly into Supabase).
    if not (OP_HOURS_START * 60 <= current_mins < OP_HOURS_END * 60):
        return (
            False,
            f"Outside lab hours ({OP_HOURS_START}:00 AM – {OP_HOURS_END - 12}:00 PM)",
            False,
            0,
        )

    with known_schedules_lock:
        fac_schedules = [s for s in known_schedules if str(s.get("faculty_id")) == str(faculty_id)]

    if not fac_schedules:
        return False, "No assigned schedule for this laboratory", False, 0

    today_schedules = [
        s for s in fac_schedules 
        if str(s.get("day_of_week", "")).strip().lower() == current_day.lower()
    ]

    if not today_schedules:
        allowed_days = ", ".join(sorted(set(str(s.get("day_of_week")).capitalize() for s in fac_schedules)))
        return False, f"No schedule today ({current_day}). Scheduled: {allowed_days}", False, 0

    for s in today_schedules:
        start_t = _normalize_time_str(s.get("start_time", "00:00:00"))
        end_t = _normalize_time_str(s.get("end_time", "23:59:59"))

        try:
            start_parts = [int(p) for p in start_t.split(":")[:2]]
            sched_start_mins = start_parts[0] * 60 + start_parts[1]
        except Exception:
            sched_start_mins = 0

        if start_t <= current_time <= end_t:
            grace_deadline_mins = sched_start_mins + LATE_GRACE_PERIOD_MINUTES
            if current_mins > grace_deadline_mins:
                mins_late = max(1, current_mins - sched_start_mins)
                return True, f"Schedule Valid ({start_t[:5]}-{end_t[:5]})", True, mins_late
            else:
                return True, f"Schedule Valid ({start_t[:5]}-{end_t[:5]})", False, 0

    window_strs = ", ".join(f"{_normalize_time_str(s.get('start_time'))[:5]}-{_normalize_time_str(s.get('end_time'))[:5]}" for s in today_schedules)
    return False, f"Outside scheduled hours ({window_strs})", False, 0


# =============================================================================
#  BACKGROUND SYNC LISTENER
# =============================================================================

def start_realtime_listener(client: Client) -> None:
    """
    Background sync daemon that periodically:
    - Pushes live face detection status to Supabase so the app can display it
    - Processes any pending face capture requests from the app
    - Refreshes the in-memory face registry
    - Syncs mode with database
    """
    global enrollment_status, enrollment_name, face_in_frame_status, cam_ready_status

    def _quick_face_check() -> tuple[bool, bool]:
        """Quick check: check global status updated by main recognition loop.
        Returns (camera_ready, face_detected).
        """
        global latest_frame, cam_ready_status, face_in_frame_status
        cam_ready = (latest_frame is not None) or cam_ready_status
        return cam_ready, face_in_frame_status

    def _push_pi_status(cam_ready: bool, face_detected: bool) -> None:
        """Write live camera + face detection state into Supabase."""
        global pi_mode
        try:
            if not LABORATORY_ID:
                return
                
            with pi_mode_lock:
                current_mode = pi_mode
                
            status_payload = {
                "camera_ready": cam_ready,
                "face_detected": face_detected,
                "mode": current_mode,
                "message": (
                    f"Mode: {current_mode.upper()} | " + (
                        "Camera Ready — Face Positioned 🟢" if (cam_ready and face_detected)
                        else ("Camera Ready — Position face in front of camera ⚠️" if cam_ready
                              else "Pi Camera Unavailable ⚠️")
                    )
                ),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            client.table("laboratories").update({
                "pi_status": status_payload,
                "mode": current_mode  # Also update the mode column
            }).eq("id", LABORATORY_ID).execute()
        except Exception as exc:
            log.debug("pi_status push skipped: %s", exc)

    def poll_forever() -> None:
        global enrollment_status, enrollment_name, pi_mode, last_decision_time
        log.info("Background profile sync & capture daemon active (polling every 2s).")
        cycle = 0
        last_pushed_tuple = None
        while True:
            time.sleep(2)
            try:
                cycle += 1
                
                # 1. Sync mode from database (every 6 seconds / 3 cycles)
                if cycle % 3 == 0:
                    sync_pi_mode_with_database(client)
                
                # 2. Check face status
                cam_ready, face_detected = _quick_face_check()
                
                # 3. Update global status (used by UI)
                cam_ready_status = cam_ready
                face_in_frame_status = face_detected
                
                # 4. Push live camera + face detection state to Supabase only on change or every 10s
                with pi_mode_lock:
                    current_mode = pi_mode
                current_status_tuple = (cam_ready, face_detected, current_mode)
                if current_status_tuple != last_pushed_tuple or cycle % 5 == 0:
                    _push_pi_status(cam_ready, face_detected)
                    last_pushed_tuple = current_status_tuple

                # 5. Process any pending face capture requests in Supabase
                # 🔥 ONLY process enrollment if in enrollment mode
                if current_mode == "enrollment":
                    res = client.table("profiles").select("*").eq("camera_status", "pending").execute()
                    if res.data and len(res.data) > 0:
                        for pending_user in res.data:
                            pid = pending_user["id"]
                            pname = pending_user.get("name", "User")
                            log.info(f"📸 Pending capture request detected for '{pname}' (ID: {pid})")

                            enrollment_status = "pending"
                            enrollment_name = pname
                            client.table("profiles").update({"camera_status": "capturing"}).eq("id", pid).execute()
                            enrollment_status = "capturing"

                            if latest_frame is not None:
                                def _mirror_progress(text: str, _pid=pid) -> None:
                                    # Lets the mobile app show the same prompt as the kiosk screen
                                    client.table("profiles").update(
                                        {"camera_status": f"capturing: {text}"}
                                    ).eq("id", _pid).execute()

                                emb, err_msg, status_code = capture_verified_live_enrollment(
                                    user_name=pname, profile_id=pid, on_progress=_mirror_progress
                                )
                                if emb is not None:
                                    client.table("profiles").update({
                                        "face_embedding": [float(v) for v in emb],
                                        "camera_status": "completed"
                                    }).eq("id", pid).execute()
                                    enrollment_status = "completed"
                                    add_history_entry(pname, "enrolled", "Face Biometrics Enrolled (live verified)")
                                    log_access(
                                        client,
                                        decision="granted",
                                        faculty_id=pid,
                                        reason=f"Enrolled: Face biometrics registered for {pname}",
                                    )
                                    last_decision_time = time.time() + 25.0  # 25s grace cooldown
                                    log.info(f"✅ Face embedding enrolled for '{pname}'!")
                                    reload_known_faces(client)
                                else:
                                    client.table("profiles").update({"camera_status": status_code}).eq("id", pid).execute()
                                    enrollment_status = status_code
                                    add_history_entry(pname, "denied", f"Enroll Rejected: {err_msg}")
                                    log.warning("⚠️ Pending capture rejected for '%s' (%s): %s", pname, status_code, err_msg)
                            else:
                                log.warning("⚠️ latest_frame is None — main loop camera not ready yet. Retrying next cycle.")
                                client.table("profiles").update({"camera_status": "failed_camera_error"}).eq("id", pid).execute()
                                enrollment_status = "failed_camera_error"
                else:
                    if enrollment_status != "idle":
                        enrollment_status = "idle"

                # 6. Periodically reload face registry every 15 cycles (30s)
                if cycle % 15 == 0:
                    reload_known_faces(client)
            except Exception as exc:
                log.warning(f"Background face sync error: {exc}")

    thread = threading.Thread(target=poll_forever, daemon=True, name="profile-sync")
    thread.start()


def start_remote_unlock_listener(client: Client) -> None:
    """
    Dedicated, high-priority background listener for remote force unlock events from Supabase.
    Runs in its own lightweight thread (polling every 0.75s) to guarantee fast < 1s response.
    """
    def _poll_unlock():
        log.info("⚡ Fast Remote Unlock listener active (polling Supabase access_logs every 0.75s).")
        last_processed_unlock_id = None

        if LABORATORY_ID:
            try:
                res = (
                    client.table("access_logs")
                    .select("id")
                    .eq("laboratory_id", LABORATORY_ID)
                    .eq("decision", "granted")
                    .or_("reason.ilike.Manual Force Unlock%,reason.ilike.Laboratory Exit%,reason.ilike.Exit%")
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                if res.data and len(res.data) > 0:
                    last_processed_unlock_id = res.data[0].get("id")
                    log.info("Initial latest unlock ID recorded: %s", last_processed_unlock_id)
            except Exception as e:
                log.debug("Initial unlock ID fetch error: %s", e)

        while True:
            time.sleep(0.75)
            if not LABORATORY_ID:
                continue
            try:
                unlock_res = (
                    client.table("access_logs")
                    .select("id, reason, created_at")
                    .eq("laboratory_id", LABORATORY_ID)
                    .eq("decision", "granted")
                    .or_("reason.ilike.Manual Force Unlock%,reason.ilike.Laboratory Exit%,reason.ilike.Exit%")
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                if unlock_res.data and len(unlock_res.data) > 0:
                    latest_unlock = unlock_res.data[0]
                    latest_id = latest_unlock.get("id")
                    if last_processed_unlock_id is None:
                        last_processed_unlock_id = latest_id
                    elif latest_id != last_processed_unlock_id:
                        last_processed_unlock_id = latest_id
                        u_reason = latest_unlock.get("reason", "Manual Force Unlock")
                        log.info(f"🔓 Supabase Remote Unlock/Exit detected: {u_reason}")
                        # Extract name if present in reason (e.g. 'Laboratory Exit: Prof. Name')
                        actor_name = "Remote App"
                        if ":" in u_reason:
                            actor_name = u_reason.split(":", 1)[1].strip()
                        add_history_entry(actor_name, "granted", u_reason)
                        unlock_door()
            except Exception as unlock_err:
                log.debug(f"Remote unlock poll error: {unlock_err}")
                time.sleep(1.0)

    t = threading.Thread(target=_poll_unlock, daemon=True, name="remote-unlock-poll")
    t.start()


# =============================================================================
#  HTTP ENROLLMENT SERVER FOR PI CAMERA
# =============================================================================

from http.server import HTTPServer, BaseHTTPRequestHandler

class EnrollmentRequestHandler(BaseHTTPRequestHandler):
    supabase_client: Client = None

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        global pi_mode, cam_ready_status, face_in_frame_status, last_detected_name
        
        if self.path in ("/status", "/health", "/"):
            display_name = last_detected_name if last_detected_name else "Unknown"
            msg = f"Mode: {pi_mode.upper()} | {'Face Detected 🟢 (' + display_name + ')' if face_in_frame_status else 'No Face ⚠️'}"
            self._send_json(200, {
                "status": "ready",
                "mode": pi_mode,
                "camera_ready": cam_ready_status,
                "face_detected": face_in_frame_status,
                "user_name": last_detected_name,
                "message": msg
            })
            return

        self.send_error(404, "Endpoint not found")

    def do_POST(self):
        global pi_mode
        
        # --- Remote Manual Unlock endpoint ---
        if self.path == "/unlock":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                reason = payload.get("reason", "Manual Force Unlock (Mobile App)")
                log.info(f"🔓 Direct HTTP /unlock received: {reason}")
                add_history_entry("Remote App", "granted", reason)
                unlock_door()
                self._send_json(200, {"success": True, "message": "Door unlocked"})
            except Exception as exc:
                log.error(f"Error handling /unlock: {exc}")
                self._send_json(500, {"success": False, "error": str(exc)})
            return

        # --- Set Pi Mode ---
        if self.path == "/mode":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            
            try:
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                new_mode = payload.get("mode", "recognition")
                
                if new_mode not in ['recognition', 'enrollment', 'idle']:
                    self._send_json(400, {"success": False, "error": "Invalid mode. Use: recognition, enrollment, idle"})
                    return
                
                # Update local mode
                with pi_mode_lock:
                    pi_mode = new_mode
                
                # Push to database
                push_pi_mode_to_database(self.supabase_client, new_mode)
                
                log.info(f"🔄 Pi mode changed to: {new_mode.upper()}")
                
                self._send_json(200, {"success": True, "mode": new_mode})
                
            except Exception as exc:
                self._send_json(500, {"success": False, "error": str(exc)})
            return
        
        # --- Enrollment endpoint ---
        if self.path == "/enroll":
            # Check if in enrollment mode
            with pi_mode_lock:
                current_mode = pi_mode
                
            if current_mode != 'enrollment':
                self._send_json(400, {
                    "success": False, 
                    "error": f"Pi is in {current_mode} mode. Set to 'enrollment' mode first."
                })
                return
            
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            
            try:
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                profile_id = payload.get("profileId") or payload.get("profile_id")
                delay = int(payload.get("delay", 0))
                
                if not profile_id:
                    self._send_json(400, {"success": False, "error": "profileId is required"})
                    return

                log.info(f"📸 Enrollment Request for profile: {profile_id} (delay={delay}s)")
                
                if delay > 0:
                    log.info(f"Waiting {delay} seconds before capture...")
                    time.sleep(delay)

                # Use cached frame from main recognition loop
                if latest_frame is None:
                    self._send_json(500, {"success": False, "error": "Camera frame not ready yet"})
                    return

                # Get profile name first so HUD can display it during live blink verification
                res = self.supabase_client.table("profiles").select("name").eq("id", profile_id).execute()
                pname = res.data[0]["name"] if (res.data and len(res.data) > 0) else "User"

                global enrollment_status, enrollment_name, last_decision_time
                enrollment_name = pname
                enrollment_status = "capturing"

                embedding, err_msg, status_code = capture_verified_live_enrollment(
                    user_name=pname, profile_id=profile_id
                )
                if embedding is None:
                    enrollment_status = status_code
                    self.supabase_client.table("profiles").update({"camera_status": status_code}).eq("id", profile_id).execute()
                    add_history_entry(pname, "denied", f"Enroll Rejected: {err_msg}")
                    self._send_json(400, {"success": False, "error": err_msg})
                    return

                emb_list = [float(v) for v in embedding]

                # Update Supabase
                self.supabase_client.table("profiles").update({
                    "face_embedding": emb_list,
                    "camera_status": "completed"
                }).eq("id", profile_id).execute()
                enrollment_status = "completed"

                # Reload local registry
                reload_known_faces(self.supabase_client)

                add_history_entry(pname, "enrolled", "Live Face Enrolled via Pi Camera")
                log_access(
                    self.supabase_client,
                    decision="granted",
                    faculty_id=profile_id,
                    reason=f"Enrolled: Face biometrics registered for {pname}",
                )
                last_decision_time = time.time() + 25.0  # 25s grace cooldown
                log.info(f"✅ ENROLLMENT SUCCESSFUL for {pname}")

                self._send_json(200, {"success": True, "name": pname})

            except Exception as exc:
                log.error(f"Error during enrollment: {exc}")
                self._send_json(500, {"success": False, "error": str(exc)})
            return

        self.send_error(404, "Endpoint not found")

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def start_enrollment_server(client: Client, port: int = 5000) -> None:
    """Start background HTTP server for app control."""
    def _run():
        EnrollmentRequestHandler.supabase_client = client
        server_address = ("", port)
        try:
            httpd = HTTPServer(server_address, EnrollmentRequestHandler)
            log.info(f"🌐 Pi Control Server listening on port {port} ...")
            log.info(f"📡 Endpoints: /status, /mode, /enroll, /unlock")
            httpd.serve_forever()
        except Exception as exc:
            log.error(f"Failed to start HTTP Server: {exc}")

    t = threading.Thread(target=_run, daemon=True, name="control-server")
    t.start()


# =============================================================================
#  GPIO / RELAY
# =============================================================================

def setup_gpio() -> None:
    """Configure GPIO pins for relay and manual override button.  Safe to call on a non-Pi host."""
    if not ON_PI:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    # Relay pin — initialise to LOCKED state
    initial = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
    GPIO.setup(RELAY_PIN, GPIO.OUT, initial=initial)
    log.info("GPIO pin %d configured (relay initialised to LOCKED).", RELAY_PIN)
    # Button pin — internal pull-up; button press pulls LOW
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    log.info("GPIO pin %d configured (manual override button, pull-up).", BUTTON_PIN)


def unlock_door() -> None:
    """
    Energise the relay to release the solenoid lock, wait DOOR_OPEN_SECONDS,
    then de-energise to lock again.  Runs in a background thread so the
    recognition loop is not blocked.
    """
    global last_decision_time
    last_decision_time = time.time()

    def _relay_pulse() -> None:
        try:
            log.info("UNLOCKING door for %d seconds ...", DOOR_OPEN_SECONDS)
            if ON_PI:
                GPIO.output(RELAY_PIN, GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH)

            time.sleep(DOOR_OPEN_SECONDS)

            if ON_PI:
                GPIO.output(RELAY_PIN, GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW)
            log.info("Door LOCKED again.")
        except Exception as exc:
            log.error("Relay pulse ERROR — door may be stuck open/closed: %s", exc)
            # Always try to re-lock the pin on failure
            try:
                if ON_PI:
                    GPIO.output(RELAY_PIN, GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW)
            except Exception:
                pass

    thread = threading.Thread(target=_relay_pulse, daemon=True, name="relay-pulse")
    thread.start()


def start_button_listener() -> None:
    """
    Background listener that watches BUTTON_PIN for a physical button press.
    Each press triggers unlock_door() — same as a face-recognition grant.
    Wiring: button leg A → GPIO 27 (Physical Pin 13), button leg B → GND.
    """
    if not ON_PI:
        log.info("Button listener skipped (not on Pi).")
        return

    def _on_button_pressed(channel: int) -> None:
        log.info("🔘 Manual override button pressed (GPIO %d) — triggering door unlock.", channel)
        add_history_entry("Manual Button", "granted", "Physical override button pressed")
        unlock_door()

    try:
        GPIO.add_event_detect(
            BUTTON_PIN,
            GPIO.FALLING,
            callback=_on_button_pressed,
            bouncetime=BUTTON_DEBOUNCE_MS,
        )
        log.info("🔘 Manual override button listener active on GPIO %d.", BUTTON_PIN)
    except Exception as exc:
        log.warning("Failed to register GPIO event detect: %s. Using polling fallback.", exc)

        def _poll() -> None:
            last_state = GPIO.HIGH
            while True:
                try:
                    state = GPIO.input(BUTTON_PIN)
                    if last_state == GPIO.HIGH and state == GPIO.LOW:
                        _on_button_pressed(BUTTON_PIN)
                        time.sleep(BUTTON_DEBOUNCE_MS / 1000.0)
                    last_state = state
                    time.sleep(0.05)
                except Exception:
                    time.sleep(0.5)

        t = threading.Thread(target=_poll, daemon=True, name="button-poll")
        t.start()


def cleanup_gpio() -> None:
    """Release GPIO resources on exit."""
    if ON_PI:
        GPIO.cleanup()
        log.info("GPIO resources released.")


# =============================================================================
#  ACCESS LOGGING
# =============================================================================

def log_access(
    client:     Client,
    decision:   str,           # "granted" or "denied"
    faculty_id: str | None,    # None for unknown faces
    reason:     str,
) -> None:
    """
    Insert a row into `access_logs`.  Runs in a background thread so the
    main recognition loop is never blocked by network latency.
    """

    def _insert() -> None:
        payload = {
            "laboratory_id": LABORATORY_ID,
            "decision":      decision,
            "reason":        reason,
        }
        if faculty_id:
            payload["faculty_id"] = faculty_id

        try:
            client.table("access_logs").insert(payload).execute()
            log.info("Access log written — decision=%s faculty_id=%s", decision, faculty_id)
        except Exception as exc:
            log.error("Failed to write access log: %s", exc)

    thread = threading.Thread(target=_insert, daemon=True, name="log-insert")
    thread.start()


# =============================================================================
#  FACE RECOGNITION & LIVENESS / ANTI-SPOOFING ENGINE
# =============================================================================

def _rank_profiles(live_norm: np.ndarray, exclude_id: str | None = None) -> list[tuple[float, str, str]]:
    """Return [(cos_dist, id, name)] sorted best-first, one entry per profile."""
    ranked: list[tuple[float, str, str]] = []
    with known_faces_lock:
        for profile in known_faces:
            if exclude_id is not None and str(profile["id"]) == str(exclude_id):
                continue
            emb = profile["embedding"]
            if emb.shape != live_norm.shape:
                continue
            ranked.append((float(1.0 - np.dot(live_norm, emb)), profile["id"], profile["name"]))
    ranked.sort(key=lambda r: r[0])
    return ranked


def find_best_match(live_embedding: np.ndarray) -> tuple[str | None, str | None, float]:
    """
    Compare a (preferably multi-frame averaged) live embedding against every
    stored profile using cosine distance.

    A match must:
      1. be below FACE_MATCH_THRESHOLD, and
      2. beat the 2nd-best profile by FACE_MATCH_MARGIN (unless it is an
         unambiguous < FACE_MATCH_SURE hit).
    Otherwise the face is treated as "not registered" — refusing is always
    better than opening the door for the wrong person.

    Returns:
        (faculty_id, name, distance) — id/name are None when no safe match.
    """
    ranked = _rank_profiles(_l2_norm(live_embedding))
    if not ranked:
        return None, None, float("inf")

    best_cos, best_id, best_name = ranked[0]
    second_cos = ranked[1][0] if len(ranked) > 1 else float("inf")

    if best_cos >= FACE_MATCH_THRESHOLD:
        return None, None, best_cos

    if best_cos > FACE_MATCH_SURE and (second_cos - best_cos) < FACE_MATCH_MARGIN:
        log.warning(
            "Ambiguous face match rejected: best=%s (%.4f) vs 2nd=%s (%.4f)",
            best_name, best_cos, ranked[1][2], second_cos,
        )
        return None, None, best_cos

    return best_id, best_name, best_cos


def find_duplicate_profile(embedding: np.ndarray, exclude_id: str | None = None) -> tuple[str | None, str | None, float]:
    """During enrollment: is this face already registered under another profile?"""
    ranked = _rank_profiles(_l2_norm(embedding), exclude_id=exclude_id)
    if ranked and ranked[0][0] < DUPLICATE_FACE_THRESHOLD:
        return ranked[0][1], ranked[0][2], ranked[0][0]
    return None, None, ranked[0][0] if ranked else float("inf")


def _scale_box_to_full(small_box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Scale a bounding box from downscaled coordinates back to full frame coordinates."""
    h, w = frame_bgr_shape = frame_shape[:2]
    top, right, bottom, left = [int(round(c / FRAME_SCALE_FACTOR)) for c in small_box]
    top = max(0, min(h - 1, top))
    bottom = max(top + 1, min(h, bottom))
    left = max(0, min(w - 1, left))
    right = max(left + 1, min(w, right))
    return (top, right, bottom, left)


def _scale_box_to_small(full_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Scale a full-frame bounding box back to downscaled coordinates for UI overlay."""
    top, right, bottom, left = [int(round(c * FRAME_SCALE_FACTOR)) for c in full_box]
    return (top, right, bottom, left)


def _detect_face_box_fast(
    frame_bgr: np.ndarray,
    prev_full_box: tuple[int, int, int, int] | None = None,
    use_roi_hint: bool = True,
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    """
    Detect the primary face bounding box on the current frame.
    When `prev_full_box` is provided and `use_roi_hint=True`, first searches a padded ROI
    around `prev_full_box` on the downscaled frame (~6ms instead of ~35ms), guaranteeing
    that `full_box` is ALWAYS freshly detected on the current frame and never stale!

    Returns:
        (full_box, small_box) or (None, None) if no face is present.
    """
    small_frame = cv2.resize(
        frame_bgr,
        (0, 0),
        fx=FRAME_SCALE_FACTOR,
        fy=FRAME_SCALE_FACTOR,
    )
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    sh, sw = rgb_small.shape[:2]

    if prev_full_box is not None and use_roi_hint:
        pt, pr, pb, pl = _scale_box_to_small(prev_full_box)
        bw = max(20, pr - pl)
        bh = max(20, pb - pt)
        pad_x = int(bw * 0.45)
        pad_y = int(bh * 0.45)
        rx0 = max(0, pl - pad_x)
        ry0 = max(0, pt - pad_y)
        rx1 = min(sw, pr + pad_x)
        ry1 = min(sh, pb + pad_y)
        roi = rgb_small[ry0:ry1, rx0:rx1]
        if roi.size > 0:
            with dlib_lock:
                roi_locs = face_recognition.face_locations(roi)
            if roi_locs:
                bt, br, bb, bl = max(roi_locs, key=lambda b: max(0, b[2] - b[0]) * max(0, b[1] - b[3]))
                best_small = (bt + ry0, br + rx0, bb + ry0, bl + rx0)
                return _scale_box_to_full(best_small, frame_bgr.shape), best_small

    with dlib_lock:
        small_locs = face_recognition.face_locations(rgb_small)
    if not small_locs:
        return None, None

    best_small = max(small_locs, key=lambda b: max(0, b[2] - b[0]) * max(0, b[1] - b[3]))
    return _scale_box_to_full(best_small, frame_bgr.shape), best_small


def _compute_eye_aspect_ratio(eye_pts: list[tuple[int, int]]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) from 6 dlib eye landmarks:
      p0 = left/outer corner, p1,p2 = upper lid, p3 = right/inner corner, p4,p5 = lower lid
    """
    if not eye_pts or len(eye_pts) < 6:
        return 0.0
    pts = np.array(eye_pts, dtype=np.float64)
    v1 = float(np.linalg.norm(pts[1] - pts[5]))
    v2 = float(np.linalg.norm(pts[2] - pts[4]))
    horiz = float(np.linalg.norm(pts[0] - pts[3]))
    if horiz < 1.0:
        return 0.0
    return (v1 + v2) / (2.0 * horiz)


# =============================================================================
#  MINIFASNET PASSIVE AI ANTI-SPOOFING (Zero-Blink Face Liveness)
# =============================================================================
MINIFASNET_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minifasnet_v2.onnx")


class MiniFASNetClassifier:
    """
    Lightweight, production-grade passive face anti-spoofing model (Silent-Face-Anti-Spoofing).
    Inspects 80x80 scaled face crop for presentation attacks:
      Class 0: Print Attack (printed photo / paper)
      Class 1: Live / Real Human Face
      Class 2: Replay Attack (phone screen / tablet / monitor playback)
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.net = None

        if os.path.exists(model_path):
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(
                    model_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                log.info("🛡️ MiniFASNet Passive AI Anti-Spoofing loaded via ONNXRuntime: %s", model_path)
            except Exception:
                try:
                    self.net = cv2.dnn.readNetFromONNX(model_path)
                    log.info("🛡️ MiniFASNet Passive AI Anti-Spoofing loaded via OpenCV DNN: %s", model_path)
                except Exception as ex:
                    log.warning("Could not load MiniFASNet model: %s", ex)

    def is_loaded(self) -> bool:
        return self.session is not None or self.net is not None

    def predict_liveness(
        self, frame_bgr: np.ndarray, full_box: tuple[int, int, int, int]
    ) -> tuple[bool, float, str, float]:
        """
        Classifies whether the face in frame_bgr is a genuine live person or a presentation attack.

        Pre-processing matches the original Silent-Face-Anti-Spoofing repo exactly:
          * crop = face box scaled by 2.7, SHIFTED to stay inside the frame (no padding)
          * resized to 80x80, BGR channel order, RAW 0-255 float values (NO /255!)
        The previous version divided by 255, which makes the model answer
        "screen replay" ~99% of the time for real faces.

        Returns:
            (is_real, confidence_pct, label, p_live)
        """
        if not self.is_loaded() or frame_bgr is None or frame_bgr.size == 0:
            return True, 100.0, "Real (model not loaded)", 1.0

        try:
            top, right, bottom, left = full_box
            box_w = float(right - left)
            box_h = float(bottom - top)
            if box_w < 30 or box_h < 30:
                return True, 50.0, "Real (face too small to judge)", 0.5

            src_h, src_w = frame_bgr.shape[:2]
            scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, 2.7)
            new_w, new_h = box_w * scale, box_h * scale
            cx, cy = left + box_w / 2.0, top + box_h / 2.0
            x1, y1 = cx - new_w / 2.0, cy - new_h / 2.0
            x2, y2 = cx + new_w / 2.0, cy + new_h / 2.0
            if x1 < 0:
                x2 -= x1
                x1 = 0
            if y1 < 0:
                y2 -= y1
                y1 = 0
            if x2 > src_w - 1:
                x1 -= x2 - src_w + 1
                x2 = src_w - 1
            if y2 > src_h - 1:
                y1 -= y2 - src_h + 1
                y2 = src_h - 1

            cropped = frame_bgr[int(y1): int(y2) + 1, int(x1): int(x2) + 1]
            if cropped.size == 0:
                return True, 50.0, "Real (empty crop)", 0.5
            img_80 = cv2.resize(cropped, (80, 80))
            blob = np.transpose(img_80.astype(np.float32), (2, 0, 1))[np.newaxis, ...]

            if self.session is not None:
                inp_name = self.session.get_inputs()[0].name
                logits = self.session.run(None, {inp_name: blob})[0].flatten()
            else:
                self.net.setInput(blob)
                logits = self.net.forward().flatten()

            exp_scores = np.exp(logits - np.max(logits))
            probs = exp_scores / np.sum(exp_scores)
            p_print, p_live, p_replay = float(probs[0]), float(probs[1]), float(probs[2])

            if p_live >= max(p_print, p_replay):
                return True, p_live * 100.0, "Live Real Human", p_live
            if p_print >= p_replay:
                return False, p_print * 100.0, "Fake Photo (Print Attack)", p_live
            return False, p_replay * 100.0, "Fake Photo (Screen Replay)", p_live
        except Exception as exc:
            log.debug("MiniFASNet inference error: %s", exc)
            return True, 50.0, "Real (inference error)", 0.5


minifasnet = MiniFASNetClassifier(MINIFASNET_MODEL_PATH)


def _check_2d_spoof_artifacts(
    frame_bgr: np.ndarray,
    full_box: tuple[int, int, int, int],
    min_size_px: int = MIN_FACE_SIZE_PX,
) -> tuple[bool, str, float]:
    """
    Inspect the face crop for 2D presentation attack (calendar / printed photo / phone screen)
    artifacts and severe motion blur:
      1. Minimum face pixel dimensions (`min_size_px`) so eyes have enough pixel resolution.
      2. Focus / sharpness check via Laplacian variance (prevents shaking a photo to blur eyes).
      3. Specular screen/glossy paper glare check (clipped white highlights inside face).
      4. Natural skin chrominance variance check in YCrCb space (rejects flat monochrome/washed prints).

    Returns:
        (is_suspect_2d, reason, laplacian_var)
    """
    h, w = frame_bgr.shape[:2]
    top, right, bottom, left = full_box
    fh = bottom - top
    fw = right - left
    if fh < min_size_px or fw < min_size_px:
        return True, "Face too small / move closer to camera", 0.0

    crop = frame_bgr[max(0, top):min(h, bottom), max(0, left):min(w, right)]
    if crop.size == 0:
        return True, "Empty face crop", 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 8.0:
        return True, "Image too blurry (hold still)", lap_var

    # Check central face region for unnatural specular screen/glossy calendar glare
    ch, cw = crop.shape[:2]
    inner = crop[int(ch * 0.15):int(ch * 0.85), int(cw * 0.15):int(cw * 0.85)]
    if inner.size > 0:
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        # Clipped bright low-saturation glare (typical of phone screen or glossy paper reflection)
        glare_mask = (hsv[:, :, 2] > 250) & (hsv[:, :, 1] < 15)
        glare_ratio = float(np.mean(glare_mask))
        if glare_ratio > 0.35:
            return True, "Screen/paper glare detected", lap_var

        ycrcb = cv2.cvtColor(inner, cv2.COLOR_BGR2YCrCb)
        cr_std = float(np.std(ycrcb[:, :, 1]))
        cb_std = float(np.std(ycrcb[:, :, 2]))
        if cr_std < 1.5 and cb_std < 1.5:
            return True, "Flat print chrominance detected", lap_var

        # Check for harsh flashlight / direct beam overexposure (blown-out white highlights)
        mean_brightness = float(np.mean(hsv[:, :, 2]))
        if mean_brightness > 228.0 or glare_ratio > 0.45:
            return True, "Too bright / harsh light (move light away)", lap_var

        # Check for OLED/LCD screen blue-backlight emission bias (strong blue excess)
        mean_b = float(np.mean(inner[:, :, 0]))
        mean_r = float(np.mean(inner[:, :, 2]))
        if mean_b > (mean_r + 38.0):
            return True, "Digital screen backlight detected", lap_var
        if lap_var > 6000.0:
            return True, "Screen Moire / print halftone artifact detected", lap_var

    return False, "", lap_var


def _extract_eye_and_cheek_patches(
    frame_bgr: np.ndarray | None,
    left_eye: list[tuple[int, int]],
    right_eye: list[tuple[int, int]],
    eye_mid: np.ndarray,
    nose_pt: np.ndarray,
    inter_ocular: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float]:
    """
    Extract canonical 20x12 grayscale patches anchored on the stable inner/outer eye corners
    (p0 and p3, which do NOT move vertically during a blink) plus a reference patch on the
    nose bridge / upper cheek.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None, None, None, 0.0

    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr

    def _crop_corner_eye(eye_pts: list[tuple[int, int]]) -> tuple[np.ndarray | None, float]:
        p0 = np.array(eye_pts[0], dtype=np.float64)
        p3 = np.array(eye_pts[3], dtype=np.float64)
        c = 0.5 * (p0 + p3)
        ew = max(8.0, float(np.linalg.norm(p3 - p0)))
        rx = max(5, int(round(0.46 * ew)))
        ry = max(3, int(round(0.32 * ew)))
        cx, cy = int(round(c[0])), int(round(c[1]))
        x0, x1 = max(0, cx - rx), min(w, cx + rx)
        y0, y1 = max(0, cy - ry), min(h, cy + ry)
        if (x1 - x0) < 4 or (y1 - y0) < 3:
            return None, 0.0
        sub = gray[y0:y1, x0:x1].astype(np.float32)
        resized = cv2.resize(sub, (20, 12), interpolation=cv2.INTER_LINEAR)
        contrast = float(np.percentile(resized, 85) - np.percentile(resized, 15))
        norm_patch = resized - float(np.mean(resized))
        return norm_patch, contrast

    l_patch, l_cont = _crop_corner_eye(left_eye)
    r_patch, r_cont = _crop_corner_eye(right_eye)
    if l_patch is None or r_patch is None:
        return None, None, None, 0.0

    ref_c = 0.55 * eye_mid + 0.45 * nose_pt
    rx_ref = max(5, int(round(0.22 * inter_ocular)))
    ry_ref = max(3, int(round(0.15 * inter_ocular)))
    rcx, rcy = int(round(ref_c[0])), int(round(ref_c[1]))
    rx0, rx1 = max(0, rcx - rx_ref), min(w, rcx + rx_ref)
    ry0, ry1 = max(0, rcy - ry_ref), min(h, rcy + ry_ref)
    if (rx1 - rx0) < 4 or (ry1 - ry0) < 3:
        return None, None, None, 0.0

    ref_sub = gray[ry0:ry1, rx0:rx1].astype(np.float32)
    ref_resized = cv2.resize(ref_sub, (20, 12), interpolation=cv2.INTER_LINEAR)
    ref_patch = ref_resized - float(np.mean(ref_resized))
    eye_contrast = 0.5 * (l_cont + r_cont)
    return l_patch, r_patch, ref_patch, eye_contrast


def _extract_face_geometry(
    landmarks: dict,
    full_box: tuple[int, int, int, int],
    lap_var: float,
    frame_bgr: np.ndarray | None = None,
) -> dict | None:
    """
    Compute 3D-aware geometric ratios AND corner-anchored photometric eye/cheek patches:
      - left_ear, right_ear, ear: raw Eye Aspect Ratios
      - pitch_ratio: vertical (eye-midpoint to nose-tip) / horizontal (inter-ocular distance)
      - ear_norm: pitch-compensated EAR (ear / pitch_ratio), invariant to tilting a 2D photo!
      - l_patch, r_patch, ref_patch, eye_contrast: pixel-level eyelid closure verification
      - parallax_3d: difference between nose-to-jaw yaw ratio and nose-bridge-to-eye yaw ratio
      - mar: Mouth Aspect Ratio
    """
    try:
        left_eye = landmarks.get("left_eye", [])
        right_eye = landmarks.get("right_eye", [])
        nose_tip = landmarks.get("nose_tip", [])
        nose_bridge = landmarks.get("nose_bridge", [])
        chin = landmarks.get("chin", [])
        top_lip = landmarks.get("top_lip", [])
        bottom_lip = landmarks.get("bottom_lip", [])

        if len(left_eye) < 6 or len(right_eye) < 6 or len(nose_tip) < 3 or len(chin) < 17:
            return None

        left_ear = _compute_eye_aspect_ratio(left_eye)
        right_ear = _compute_eye_aspect_ratio(right_eye)
        ear = 0.5 * (left_ear + right_ear)

        l_center = np.mean(np.array(left_eye, dtype=np.float64), axis=0)
        r_center = np.mean(np.array(right_eye, dtype=np.float64), axis=0)
        eye_mid = 0.5 * (l_center + r_center)
        inter_ocular = float(np.linalg.norm(r_center - l_center))
        if inter_ocular < 18.0:
            return None

        nose_pt = np.array(nose_tip[2], dtype=np.float64)
        nose_dist = float(np.linalg.norm(nose_pt - eye_mid))
        pitch_ratio = max(0.30, nose_dist / inter_ocular)

        # 3D Head Yaw & Pitch angles
        # Center = 0.0, Turning Left < -0.07, Turning Right > +0.07
        head_yaw = float((nose_pt[0] - eye_mid[0]) / max(1.0, inter_ocular))
        head_pitch = float((nose_pt[1] - eye_mid[1]) / max(1.0, inter_ocular))

        # Pitch-compensated EAR: tilting a flat calendar photo scales both `ear` and `pitch_ratio`
        # by cos(theta), so `ear_norm` stays flat on tilted photos but drops sharply on real blinks!
        ear_norm = ear / pitch_ratio

        # 3D out-of-plane yaw parallax (nose tip protrudes in front of jaw plane)
        jaw_l = np.array(chin[1], dtype=np.float64)
        jaw_r = np.array(chin[15], dtype=np.float64)
        d_jl = float(np.linalg.norm(nose_pt - jaw_l))
        d_jr = float(np.linalg.norm(nose_pt - jaw_r))
        jaw_yaw = (d_jl - d_jr) / max(1.0, d_jl + d_jr)

        bridge_pt = np.array(nose_bridge[0], dtype=np.float64) if nose_bridge else eye_mid
        d_el = float(np.linalg.norm(bridge_pt - np.array(left_eye[0], dtype=np.float64)))
        d_er = float(np.linalg.norm(bridge_pt - np.array(right_eye[3], dtype=np.float64)))
        eye_yaw = (d_el - d_er) / max(1.0, d_el + d_er)
        parallax_3d = jaw_yaw - eye_yaw

        mar = 0.0
        if len(top_lip) >= 10 and len(bottom_lip) >= 10:
            m_w = float(np.linalg.norm(np.array(top_lip[6]) - np.array(top_lip[0])))
            m_h = float(np.linalg.norm(np.array(top_lip[9]) - np.array(bottom_lip[9])))
            if m_w > 1.0:
                mar = (m_h / m_w) / pitch_ratio

        l_patch, r_patch, ref_patch, eye_contrast = _extract_eye_and_cheek_patches(
            frame_bgr, left_eye, right_eye, eye_mid, nose_pt, inter_ocular
        )

        # Extract eye crops for MRL Eye CNN Classifier (64x64)
        eye_side = max(24.0, inter_ocular * 0.70)
        half = eye_side * 0.5
        h_f, w_f = frame_bgr.shape[:2]

        l_x1 = int(max(0, l_center[0] - half))
        l_y1 = int(max(0, l_center[1] - half))
        l_x2 = int(min(w_f, l_center[0] + half))
        l_y2 = int(min(h_f, l_center[1] + half))
        l_crop = frame_bgr[l_y1:l_y2, l_x1:l_x2]

        r_x1 = int(max(0, r_center[0] - half))
        r_y1 = int(max(0, r_center[1] - half))
        r_x2 = int(min(w_f, r_center[0] + half))
        r_y2 = int(min(h_f, r_center[1] + half))
        r_crop = frame_bgr[r_y1:r_y2, r_x1:r_x2]

        eye_closed_prob = 0.0

        return {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "ear": ear,
            "pitch_ratio": pitch_ratio,
            "ear_norm": ear_norm,
            "parallax_3d": parallax_3d,
            "mar": mar,
            "center": eye_mid,
            "inter_ocular": inter_ocular,
            "full_box": full_box,
            "lap_var": lap_var,
            "l_patch": l_patch,
            "r_patch": r_patch,
            "ref_patch": ref_patch,
            "eye_contrast": eye_contrast,
            "eye_closed_prob": eye_closed_prob,
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
        }
    except Exception:
        return None


def _face_box_usable(full_box: tuple[int, int, int, int], frame_shape: tuple[int, ...], min_size_px: int) -> tuple[bool, str]:
    """Face must be big enough and fully inside the frame (cut-off faces give garbage landmarks)."""
    h, w = frame_shape[:2]
    top, right, bottom, left = full_box
    if (bottom - top) < min_size_px or (right - left) < min_size_px:
        return False, "Move closer to the camera"
    margin = 6
    if top < margin or left < margin or bottom > h - margin or right > w - margin:
        return False, "Center your face in the frame"
    return True, ""


def _passive_spoof_verdict(scores: list[float]) -> tuple[bool, float]:
    """(is_spoof, mean_p_live) from the MiniFASNet scores collected so far."""
    if not PASSIVE_ANTISPOOF or not minifasnet.is_loaded() or len(scores) < PASSIVE_MIN_FRAMES:
        return False, 1.0
    mean_live = float(np.mean(scores[-20:]))
    return mean_live < PASSIVE_LIVE_MIN, mean_live


def capture_verified_live_enrollment(
    user_name: str = "User",
    profile_id: str | None = None,
    on_progress=None,
    timeout_sec: float = ENROLLMENT_TIMEOUT_SECONDS,
    frame_source=None,
) -> tuple[np.ndarray | None, str, str]:
    """
    Liveness-verified enrollment capture.

    Flow shown to the person in front of the Pi:
      1. "Look straight at the camera"           (calibrates the straight pose)
      2. ENROLLMENT_CHALLENGES random prompts    (LEFT / RIGHT / UP / DOWN),
         each followed by "Look straight" again. Every turn must show real 3D
         nose parallax — a photo or phone screen moves as a flat plane and fails.
      3. Frontal embeddings are collected only while looking straight, checked
         for consistency (same person the whole time) and averaged.
      4. Rejected if the face is already registered to ANOTHER profile.

    Args:
        on_progress:  optional callable(str) — receives the current prompt, used
                      to mirror instructions into Supabase so the app can show them.
        frame_source: optional callable() -> BGR frame (defaults to latest_frame).

    Returns:
        (embedding_or_None, human_message, camera_status_code)
        camera_status_code is one of: completed | failed_no_face |
        failed_spoof: <why> | failed_duplicate: <name> | failed_timeout: <why>
    """
    global active_face_locations, active_face_label, active_box_color
    global face_in_frame_status, enrollment_capture_active

    enrollment_capture_active = True
    session = ChallengeSession(
        num_challenges=ENROLLMENT_CHALLENGES,
        directions=(LIVENESS_LEFT, LIVENESS_RIGHT, LIVENESS_UP, LIVENESS_DOWN),
        require_recenter=True,
        timeout_s=timeout_sec + 5.0,  # the outer loop owns the timeout
    )
    frontal_embs: list[np.ndarray] = []
    passive_scores: list[float] = []
    last_emb_time = 0.0
    any_face_seen = False
    prev_full_box = None
    frame_counter = 0
    last_progress = None
    start_t = time.time()

    def _progress(text: str) -> None:
        nonlocal last_progress
        if text == last_progress:
            return
        last_progress = text
        if on_progress is not None:
            try:
                on_progress(text)
            except Exception as exc:
                log.debug("enrollment progress callback failed: %s", exc)

    def _fail(msg: str, code: str) -> tuple[None, str, str]:
        global active_box_color, active_face_label
        active_box_color = (0, 0, 255)
        active_face_label = f"ENROLL FAILED: {msg}"
        log.warning("🚫 Enrollment rejected for '%s': %s", user_name, msg)
        return None, msg, code

    log.info("[ENROLL] Live challenge enrollment started for '%s' (%d random prompts, timeout %.0fs)",
             user_name, ENROLLMENT_CHALLENGES, timeout_sec)
    _progress("Look straight at the camera")

    try:
        while (time.time() - start_t) < timeout_sec:
            frame = frame_source() if frame_source is not None else latest_frame
            if frame is None or frame.size == 0:
                time.sleep(0.04)
                continue

            now = time.time()
            frame_counter += 1
            frame_copy = frame.copy()
            rgb_full = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

            full_box, best_small = _detect_face_box_fast(
                frame_copy, prev_full_box=prev_full_box, use_roi_hint=(prev_full_box is not None and frame_counter % 4 != 0)
            )
            if full_box is None or best_small is None:
                face_in_frame_status = False
                active_face_locations = []
                prev_full_box = None
                active_face_label = f"ENROLLING {user_name}: Position face in frame"
                active_box_color = (0, 165, 255)
                _progress("Position your face in front of the camera")
                time.sleep(0.04)
                continue

            prev_full_box = full_box
            active_face_locations = [best_small]
            face_in_frame_status = True

            usable, why = _face_box_usable(full_box, frame_copy.shape, MIN_ENROLL_FACE_SIZE_PX)
            if not usable:
                active_face_label = f"ENROLLING {user_name}: {why}"
                active_box_color = (0, 165, 255)
                _progress(why)
                time.sleep(0.04)
                continue
            any_face_seen = True

            with dlib_lock:
                lm_list = face_recognition.face_landmarks(rgb_full, [full_box])
            pts = landmarks_to_array(lm_list[0]) if lm_list else None
            if pts is None:
                time.sleep(0.03)
                continue

            res = session.update(pts, now=now)

            if PASSIVE_ANTISPOOF and minifasnet.is_loaded() and frame_counter % 2 == 0:
                _, _, _, p_live = minifasnet.predict_liveness(frame_copy, full_box)
                passive_scores.append(p_live)

            if res.state == "failed":
                return _fail(res.prompt, f"failed_spoof: {res.prompt}")

            # Frontal samples: only while looking straight (after calibration)
            if session.is_centered(pts) and (now - last_emb_time) >= 0.25 and len(frontal_embs) < ENROLLMENT_SAMPLES_REQUIRED + 4:
                with dlib_lock:
                    encs = face_recognition.face_encodings(rgb_full, [full_box], num_jitters=2, model="large")
                if encs:
                    frontal_embs.append(_l2_norm(np.array(encs[0], dtype=np.float64)))
                    last_emb_time = now

            if res.state == "passed":
                if len(frontal_embs) < ENROLLMENT_SAMPLES_REQUIRED:
                    active_box_color = (0, 255, 180)
                    active_face_label = f"ENROLLING {user_name}: Look straight — capturing ({len(frontal_embs)}/{ENROLLMENT_SAMPLES_REQUIRED})"
                    _progress("Look straight at the camera (capturing)")
                    time.sleep(0.03)
                    continue

                is_spoof, mean_live = _passive_spoof_verdict(passive_scores)
                if is_spoof:
                    return _fail("Photo/screen detected by anti-spoof model",
                                 f"failed_spoof: Photo/screen detected (live score {mean_live:.2f})")

                # Same person the whole time? Drop outliers, need >= 3 consistent samples
                stack = np.stack(frontal_embs, axis=0)
                mean_emb = _l2_norm(np.mean(stack, axis=0))
                dists = 1.0 - stack @ mean_emb
                consistent = stack[dists <= 0.030]
                if len(consistent) < 3:
                    return _fail("Face changed during capture — one person only",
                                 "failed_spoof: Face changed during capture (one person only)")
                mean_emb = _l2_norm(np.mean(consistent, axis=0))

                dup_id, dup_name, dup_dist = find_duplicate_profile(mean_emb, exclude_id=profile_id)
                if dup_id is not None:
                    return _fail(f"Face already registered as {dup_name}",
                                 f"failed_duplicate: Face already registered as {dup_name}")

                active_box_color = (0, 255, 0)
                active_face_label = f"ENROLLED: {user_name}"
                log.info("[ENROLL] '%s' enrolled — %d samples, nearest other profile %.4f, passive live %.2f",
                         user_name, len(consistent), dup_dist, _passive_spoof_verdict(passive_scores)[1])
                _progress("Done — face registered")
                return mean_emb, "completed", "completed"

            done, total = res.progress
            active_box_color = (0, 255, 200) if res.state != "calibrating" else (255, 200, 0)
            active_face_label = f"ENROLLING {user_name} [{done}/{total}]: {res.prompt}"
            _progress(f"{res.prompt} ({done}/{total})")
            time.sleep(0.02)

        if not any_face_seen:
            return _fail("No face detected by camera", "failed_no_face")
        return _fail("Timed out — follow the prompts on the screen",
                     "failed_timeout: Follow the on-screen prompts (turn your head as asked)")
    finally:
        enrollment_capture_active = False


def is_within_operating_hours() -> bool:
    """
    Return True if the current local time falls within operating hours:
      - Weekday: Monday-Saturday (Sunday is excluded)
      - Hour   : 7:00 AM up to (but not including) 5:00 PM

    Uses the Pi's local system clock, so make sure the Pi's timezone is
    set correctly (e.g. `sudo timedatectl set-timezone Asia/Manila`).
    """
    now = time.localtime()
    return now.tm_wday in OP_DAYS and OP_HOURS_START <= now.tm_hour < OP_HOURS_END


# =============================================================================
#  UI DASHBOARD HUD OVERLAY (For VNC / Local Monitor)
# =============================================================================

current_status_label: str = "SYSTEM ONLINE — READY"
current_status_color: tuple[int, int, int] = (0, 255, 0)
last_event_text: str = "System initialised. Waiting for face scan..."
last_event_color: tuple[int, int, int] = (200, 200, 200)

# Global access activity log history (list of dicts)
access_history: list[dict] = []
max_history_items: int = 5


def add_history_entry(name: str, decision: str, reason: str) -> None:
    """Record a scan result to the live activity feed display."""
    global last_event_text, last_event_color, last_decision_time
    t_str = time.strftime("%I:%M:%S %p")
    entry = {
        "time": t_str,
        "name": name,
        "decision": decision,  # 'granted' or 'denied'
        "reason": reason,
    }
    access_history.insert(0, entry)
    if len(access_history) > max_history_items:
        access_history.pop()

    if decision == "late":
        last_decision_time = time.time()
        last_event_text = f"GRANTED (LATE): {name} - Door Unlocked"
        last_event_color = (255, 165, 0)  # Orange / Amber
    elif decision == "granted":
        last_decision_time = time.time()
        if "exit" in reason.lower():
            last_event_text = f"EXIT: {name} - Door Unlocked"
            last_event_color = (59, 130, 246)  # Electric Blue for Exit
        else:
            last_event_text = f"GRANTED: {name} - Door Unlocked"
            last_event_color = (0, 255, 120)
    elif decision == "enrolled":
        last_event_text = f"ENROLLED: {name} - Face Biometrics Registered"
        last_event_color = (139, 92, 246)  # Sleek Purple for Enrollment
    elif "Outside" in reason:
        last_event_text = f"DENIED: {name} — Outside Operating Hours"
        last_event_color = (255, 165, 0)
    else:
        last_event_text = f"DENIED: {name} — Access Rejected"
        last_event_color = (255, 60, 60)


# =============================================================================
#  PYGAME HIGH-RESOLUTION NATIVE KIOSK UI DASHBOARD
# =============================================================================

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    log.warning("Pygame not installed in environment — using fallback renderer.")


def render_pygame_kiosk(
    screen: "pygame.Surface",
    fonts: dict,
    camera_surface: "pygame.Surface" = None,
    face_locations: list = None,
    face_label: str = "",
    box_color: tuple[int, int, int] = (0, 255, 0),
    active_idx: int = 0,
) -> None:
    """
    Render a high-resolution 1024x600 Pygame Kiosk Dashboard on the Pi display.
    """
    global face_in_frame_status, cam_ready_status, enrollment_status, enrollment_name, pi_mode
    
    if not PYGAME_AVAILABLE:
        return

    # Color Palette (Dark Theme Slate)
    BG_COLOR = (11, 15, 25)
    HEADER_BG = (30, 41, 59)
    CARD_BG = (30, 41, 59)
    CARD_BORDER = (51, 65, 85)
    TEXT_MAIN = (248, 250, 252)
    TEXT_MUTED = (148, 163, 184)
    ACCENT_BLUE = (59, 130, 246)
    GREEN = (16, 185, 129)
    RED = (239, 68, 68)
    ORANGE = (245, 158, 11)

    screen.fill(BG_COLOR)
    win_w, win_h = screen.get_size()

    # --- 1. TOP HEADER BAR ---
    pygame.draw.rect(screen, HEADER_BG, (0, 0, win_w, 60))
    pygame.draw.line(screen, ACCENT_BLUE, (0, 60), (win_w, 60), 2)

    # Header Title
    title_surf = fonts["title"].render("LAB ACCESS CONTROL KIOSK", True, TEXT_MAIN)
    sub_surf = fonts["sub"].render("|   CCS LABORATORY", True, ACCENT_BLUE)
    screen.blit(title_surf, (20, 16))
    screen.blit(sub_surf, (title_surf.get_width() + 30, 18))

    # Realtime Clock & Status Badge (Top Right)
    t_str = time.strftime("%I:%M:%S %p   %b %d, %Y")
    clock_surf = fonts["bold"].render(t_str, True, (250, 204, 21))
    screen.blit(clock_surf, (win_w - clock_surf.get_width() - 20, 18))

    # --- 2. LEFT PANEL: CAMERA VIEWPORT (640x480) ---
    cam_x, cam_y = 20, 80
    cam_w, cam_h = 640, 480

    # Draw Card Background & Border for Camera
    pygame.draw.rect(screen, CARD_BG, (cam_x, cam_y, cam_w, cam_h), border_radius=8)
    pygame.draw.rect(screen, CARD_BORDER, (cam_x, cam_y, cam_w, cam_h), 2, border_radius=8)

    if camera_surface is not None:
        # Blit live camera feed (direct blit if matching size, or fast scale)
        if camera_surface.get_size() == (cam_w, cam_h):
            screen.blit(camera_surface, (cam_x, cam_y))
        else:
            scaled_cam = pygame.transform.scale(camera_surface, (cam_w, cam_h))
            screen.blit(scaled_cam, (cam_x, cam_y))

        # Render Bounding Box & Target Reticle
        if face_locations and len(face_locations) > 0:
            scale_x = cam_w / 640.0
            scale_y = cam_h / 480.0

            top, right, bottom, left = [
                int(c / FRAME_SCALE_FACTOR) for c in face_locations[0]
            ]
            bx = cam_x + int(left * scale_x)
            by = cam_y + int(top * scale_y)
            bw = int((right - left) * scale_x)
            bh = int((bottom - top) * scale_y)

            # Pygame Box Color
            pg_col = (box_color[2], box_color[1], box_color[0])  # BGR -> RGB

            # Corner brackets
            clen = max(15, min(30, bw // 4))
            # Box rect
            pygame.draw.rect(screen, pg_col, (bx, by, bw, bh), 2)

            # Reticle L-Corners
            pygame.draw.line(screen, pg_col, (bx, by), (bx + clen, by), 4)
            pygame.draw.line(screen, pg_col, (bx, by), (bx, by + clen), 4)
            pygame.draw.line(screen, pg_col, (bx + bw, by), (bx + bw - clen, by), 4)
            pygame.draw.line(screen, pg_col, (bx + bw, by), (bx + bw, by + clen), 4)
            pygame.draw.line(screen, pg_col, (bx, by + bh), (bx + clen, by + bh), 4)
            pygame.draw.line(screen, pg_col, (bx, by + bh), (bx, by + bh - clen), 4)
            pygame.draw.line(screen, pg_col, (bx + bw, by + bh), (bx + bw - clen, by + bh), 4)
            pygame.draw.line(screen, pg_col, (bx + bw, by + bh), (bx + bw, by + bh - clen), 4)

            # Label Tag
            if face_label:
                lbl_surf = fonts["small_bold"].render(face_label, True, (0, 0, 0))
                tag_w = lbl_surf.get_width() + 16
                pygame.draw.rect(screen, pg_col, (bx, max(cam_y, by - 26), tag_w, 24), border_radius=4)
                screen.blit(lbl_surf, (bx + 8, max(cam_y + 2, by - 24)))
    else:
        # Camera connecting standby message inside card
        msg_surf = fonts["bold"].render("SEARCHING FOR CAMERA STREAM...", True, ORANGE)
        sub_msg = fonts["small"].render(f"Checking /dev/video{active_idx} (V4L2 / libcamera)", True, TEXT_MUTED)
        screen.blit(msg_surf, (cam_x + 130, cam_y + 210))
        screen.blit(sub_msg, (cam_x + 150, cam_y + 245))

    # --- 3. RIGHT PANEL: DASHBOARD CONTROL WIDGETS ---
    right_x = 680
    right_w = win_w - right_x - 20

    # --- Widget A: Door Lock Status Card ---
    door_locked = not (time.time() - last_decision_time < DOOR_OPEN_SECONDS and "GRANTED" in last_event_text)
    lock_bg = (127, 29, 29) if door_locked else (6, 78, 59)
    lock_border = RED if door_locked else GREEN
    lock_txt = "LOCKED  [SOLENOID SECURE]" if door_locked else "UNLOCKED  [ACCESS GRANTED]"
    lock_col = (254, 202, 202) if door_locked else (167, 243, 208)

    pygame.draw.rect(screen, lock_bg, (right_x, 80, right_w, 75), border_radius=8)
    pygame.draw.rect(screen, lock_border, (right_x, 80, right_w, 75), 2, border_radius=8)

    door_hdr = fonts["small"].render("DOOR RELAY CONTROL (GPIO 18)", True, lock_col)
    door_val = fonts["bold"].render(lock_txt, True, lock_col)
    screen.blit(door_hdr, (right_x + 16, 90))
    screen.blit(door_val, (right_x + 16, 112))

    # --- Widget B: System Telemetry Card ---
    pygame.draw.rect(screen, CARD_BG, (right_x, 170, right_w, 75), border_radius=8)
    pygame.draw.rect(screen, CARD_BORDER, (right_x, 170, right_w, 75), 1, border_radius=8)

    t_hdr = fonts["sub"].render("SYSTEM TELEMETRY", True, TEXT_MAIN)
    screen.blit(t_hdr, (right_x + 16, 182))

    t1 = fonts["small"].render(f"Enrolled Profiles: {len(known_faces)} faces   |   Camera: /dev/video{active_idx}", True, TEXT_MUTED)
    t2 = fonts["small"].render(f"Operating Hours:  7:00 AM - 7:00 PM  (Mon - Sat)", True, TEXT_MUTED)
    screen.blit(t1, (right_x + 16, 210))
    screen.blit(t2, (right_x + 16, 228))

    # --- Widget C: Pi Camera & Enrollment Status Card ---
    pygame.draw.rect(screen, CARD_BG, (right_x, 258, right_w, 120), border_radius=8)
    pygame.draw.rect(screen, CARD_BORDER, (right_x, 258, right_w, 120), 1, border_radius=8)

    cs_hdr = fonts["sub"].render("PI CAMERA & ENROLLMENT", True, TEXT_MAIN)
    screen.blit(cs_hdr, (right_x + 16, 270))

    # Camera ready row
    cr_col = GREEN if cam_ready_status else ORANGE
    cr_lbl = fonts["small"].render("Camera:", True, TEXT_MUTED)
    cr_val = fonts["small_bold"].render("READY" if cam_ready_status else "UNAVAILABLE", True, cr_col)
    screen.blit(cr_lbl, (right_x + 16, 296))
    screen.blit(cr_val, (right_x + 100, 296))

    # Face detected row
    if face_in_frame_status:
        if last_detected_name and last_detected_name not in ("Unknown Face", "Face Not Found"):
            fd_col = GREEN
            fd_txt = f"DETECTED — {last_detected_name}"
        elif last_detected_name in ("Unknown Face", "Face Not Found"):
            fd_col = ORANGE
            fd_txt = "DETECTED — Face Not Found"
        else:
            fd_col = GREEN
            fd_txt = "DETECTED IN VIEW"
    else:
        fd_col = TEXT_MUTED
        fd_txt = "NOT DETECTED"
    fd_lbl = fonts["small"].render("Face:", True, TEXT_MUTED)
    fd_val = fonts["small_bold"].render(fd_txt[:32], True, fd_col)
    screen.blit(fd_lbl, (right_x + 16, 315))
    screen.blit(fd_val, (right_x + 100, 315))

    # 🔥 NEW: Mode display row
    mode_display_colors = {
        'recognition': GREEN,
        'enrollment': ORANGE,
        'idle': TEXT_MUTED
    }
    mode_display_color = mode_display_colors.get(pi_mode, TEXT_MUTED)
    mode_lbl = fonts["small"].render("Mode:", True, TEXT_MUTED)
    mode_val = fonts["small_bold"].render(pi_mode.upper(), True, mode_display_color)
    screen.blit(mode_lbl, (right_x + 16, 334))
    screen.blit(mode_val, (right_x + 100, 334))

    # Enrollment status row
    enroll_colors = {
        "idle":          TEXT_MUTED,
        "pending":       ORANGE,
        "capturing":     (59, 130, 246),
        "completed":     GREEN,
        "failed_spoof":     RED,
        "failed_no_face":   RED,
        "failed_camera_error": RED,
        "failed_camera_offline": RED,
    }
    enroll_labels = {
        "idle":          "IDLE - Awaiting app request",
        "pending":       f"PENDING - {enrollment_name or 'User'}",
        "capturing":     f"ENROLLING - {enrollment_name or 'User'}",
        "completed":     f"DONE - {enrollment_name or 'User'} Enrolled!",
        "failed_spoof":     "BLOCKED - Photo/Spoof Rejected!",
        "failed_no_face":   "FAILED - No live face detected",
        "failed_camera_error": "FAILED - Camera error",
        "failed_camera_offline": "FAILED - Camera offline",
    }
    if enrollment_status.startswith("failed_spoof"):
        en_col = RED
        reason_part = enrollment_status.replace("failed_spoof:", "").strip()
        en_txt = f"BLOCKED - {reason_part}" if reason_part else "BLOCKED - Spoof Rejected"
    else:
        en_col = enroll_colors.get(enrollment_status, TEXT_MUTED)
        en_txt = enroll_labels.get(enrollment_status, enrollment_status.upper())
    en_lbl = fonts["small"].render("Enroll:", True, TEXT_MUTED)
    en_val = fonts["small_bold"].render(en_txt[:26], True, en_col)
    screen.blit(en_lbl, (right_x + 16, 353))
    screen.blit(en_val, (right_x + 100, 353))

    # --- Widget D: Live Activity Feed Table ---
    pygame.draw.rect(screen, CARD_BG, (right_x, 393, right_w, 167), border_radius=8)
    pygame.draw.rect(screen, CARD_BORDER, (right_x, 393, right_w, 167), 1, border_radius=8)

    act_hdr = fonts["sub"].render("LIVE SCAN LOGS", True, TEXT_MAIN)
    screen.blit(act_hdr, (right_x + 16, 405))

    if not access_history:
        no_logs = fonts["small"].render("Waiting for face detection...", True, TEXT_MUTED)
        screen.blit(no_logs, (right_x + 16, 437))
    else:
        y_off = 435
        for item in access_history[:4]:
            if item["decision"] == "granted":
                if "exit" in item.get("reason", "").lower():
                    tag_col = (59, 130, 246)  # Electric Blue for Exit
                    dec_text = "EXIT"
                else:
                    tag_col = GREEN
                    dec_text = "GRANTED"
            elif item["decision"] == "enrolled":
                tag_col = (59, 130, 246)  # Electric Blue for Enrolled
                dec_text = "ENROLLED"
            elif item["decision"] == "late":
                tag_col = ORANGE
                dec_text = "LATE"
            elif "Outside" in item["reason"]:
                tag_col = ORANGE
                dec_text = "DENIED"
            else:
                tag_col = RED
                dec_text = "DENIED"

            t_surf = fonts["small"].render(item["time"], True, TEXT_MUTED)
            n_surf = fonts["small_bold"].render(item["name"][:18], True, TEXT_MAIN)
            d_surf = fonts["small"].render(dec_text, True, tag_col)

            screen.blit(t_surf, (right_x + 16, y_off))
            screen.blit(n_surf, (right_x + 105, y_off))
            screen.blit(d_surf, (right_x + 235, y_off))
            y_off += 32

    pygame.display.flip()


def draw_dashboard_overlay(
    frame: np.ndarray,
    face_locations: list = None,
    face_label: str = "",
    box_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw a premium, Tesla/Cyberpunk high-tech security kiosk dashboard.
    Includes frosted glass top header, live telemetry side cards, neon reticles, and bottom ticker.
    """
    global face_in_frame_status, pi_mode
    
    h, w = frame.shape[:2]

    # --- 1. Semi-transparent Frosted Overlay Backgrounds ---
    overlay = frame.copy()

    # Top Header Bar (56px high) - Dark Navy Slate
    cv2.rectangle(overlay, (0, 0), (w, 56), (15, 20, 28), -1)

    # Bottom Ticker Bar (40px high)
    cv2.rectangle(overlay, (0, h - 40), (w, h), (15, 20, 28), -1)

    # Top-Right Telemetry Card Panel (Width 230px, Height 115px)
    card_w = 230
    cv2.rectangle(overlay, (w - card_w - 10, 65), (w - 10, 175), (15, 20, 28), -1)

    # Apply 70% opacity blend
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Accent Underline for Header (Electric Blue Line)
    cv2.line(frame, (0, 56), (w, 56), (235, 150, 20), 2, cv2.LINE_AA)

    # Border outline for Top-Right Telemetry Card
    cv2.rectangle(frame, (w - card_w - 10, 65), (w - 10, 175), (60, 75, 95), 1, cv2.LINE_AA)

    # --- 2. Header Text & Indicators ---
    # App Title & Lab Name
    cv2.putText(
        frame,
        "LAB ACCESS CONTROL  |  CCS LABORATORY",
        (15, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Real-time Clock & Date
    time_str = time.strftime("%I:%M:%S %p")
    cv2.putText(
        frame,
        time_str,
        (w - 165, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 220, 100),
        2,
        cv2.LINE_AA,
    )

    # Pulsing Live Dot (Neon Green)
    cv2.circle(frame, (w - 180, 30), 5, (0, 255, 120), -1, cv2.LINE_AA)

    # --- 3. Telemetry Card Content (Top Right) ---
    door_locked = not (time.time() - last_decision_time < DOOR_OPEN_SECONDS and "GRANTED" in last_event_text)
    lock_str = "LOCKED [GPIO 18]" if door_locked else "UNLOCKED [ACTIVE]"
    lock_col = (50, 50, 235) if door_locked else (0, 255, 120)

    user_display = last_detected_name if last_detected_name else ("None in view" if not face_in_frame_status else "Scanning...")
    user_col = (0, 255, 120) if (last_detected_name and last_detected_name not in ("Unknown Face", "Face Not Found")) else ((0, 165, 255) if last_detected_name in ("Unknown Face", "Face Not Found") else (180, 180, 180))

    cv2.putText(frame, "SYSTEM TELEMETRY", (w - card_w, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"DOOR: {lock_str}", (w - card_w, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.48, lock_col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"USER: {user_display[:20]}", (w - card_w, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.48, user_col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FACES LOADED: {len(known_faces)} profiles", (w - card_w, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"HOURS: 7 AM - 7 PM (Mon-Sat)", (w - card_w, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)

    # --- 4. Bottom Activity Ticker ---
    cv2.putText(
        frame,
        f"LAST SCAN: {last_event_text}",
        (15, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        last_event_color,
        1,
        cv2.LINE_AA,
    )

    # --- 5. Cyberpunk HUD Target Reticle around Detected Face ---
    if face_locations and len(face_locations) > 0:
        top, right, bottom, left = [
            int(coord / FRAME_SCALE_FACTOR) for coord in face_locations[0]
        ]

        # Inner Bounding Rectangle (Thinner)
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 1, cv2.LINE_AA)

        # Tech L-Corner Reticles (Thicker 3px)
        c_len = max(15, min(30, (right - left) // 4))
        # Top-Left
        cv2.line(frame, (left, top), (left + c_len, top), box_color, 3)
        cv2.line(frame, (left, top), (left, top + c_len), box_color, 3)
        # Top-Right
        cv2.line(frame, (right, top), (right - c_len, top), box_color, 3)
        cv2.line(frame, (right, top), (right, top + c_len), box_color, 3)
        # Bottom-Left
        cv2.line(frame, (left, bottom), (left + c_len, bottom), box_color, 3)
        cv2.line(frame, (left, bottom), (left, bottom - c_len), box_color, 3)
        # Bottom-Right
        cv2.line(frame, (right, bottom), (right - c_len, bottom), box_color, 3)
        cv2.line(frame, (right, bottom), (right, bottom - c_len), box_color, 3)

        # Center Crosshair Dot
        cx, cy = (left + right) // 2, (top + bottom) // 2
        cv2.circle(frame, (cx, cy), 3, box_color, -1, cv2.LINE_AA)

        # Cyberpunk Label Tag Above Face Box
        if face_label:
            (tw, th), _ = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(
                frame,
                (left, top - th - 12),
                (left + tw + 14, top),
                box_color,
                -1,
            )
            cv2.putText(
                frame,
                face_label,
                (left + 7, top - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (10, 10, 10),
                2,
                cv2.LINE_AA,
            )

    return frame


def safe_show_frame(window_name: str, frame: np.ndarray) -> bool:
    """
    Safely display OpenCV preview window if a GUI display is active.
    Returns True if user pressed 'q' to quit.
    """
    if os.environ.get("HEADLESS", "0") == "1":
        time.sleep(0.01)
        return False
    try:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return True
    except cv2.error:
        time.sleep(0.01)
    except Exception:
        time.sleep(0.01)
    return False


# Try importing Picamera2 (Official Raspberry Pi Foundation CSI Ribbon Camera Library)
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except Exception:
    HAS_PICAMERA2 = False


class Picamera2Adapter:
    """Wrapper around Picamera2 to mimic OpenCV VideoCapture interface for Pi Ribbon Cameras."""
    def __init__(self) -> None:
        self.picam2 = Picamera2()
        # NOTE: Picamera2 names formats by register order. "RGB888" is stored as
        # B,G,R in memory = what OpenCV expects. ("BGR888" gives R,G,B -> blue faces,
        # which hurts recognition and makes the anti-spoof model flag real people.)
        config = self.picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        self.picam2.configure(config)
        self.picam2.start()
        log.info("Picamera2 CSI Ribbon Camera initialized at 640x480 (30 FPS).")

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        try:
            frame = self.picam2.capture_array()
            if frame is not None and frame.size > 0:
                return True, frame
        except Exception as exc:
            log.warning("Picamera2 frame capture warning: %s", exc)
        return False, None

    def release(self) -> None:
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception:
            pass


def smart_open_camera() -> tuple[object, int]:
    """
    Open camera stream cleanly without backend collisions or unreleased handles.
    """
    preferred = int(os.environ.get("CAMERA_INDEX", CAMERA_INDEX))

    # 1. Try Native Picamera2 only if explicitly requested via environment variable
    if os.environ.get("USE_PICAMERA2", "0") == "1" and HAS_PICAMERA2:
        try:
            log.info("Attempting to open Pi Camera via native Picamera2 ...")
            adapter = Picamera2Adapter()
            ret, test_frame = adapter.read()
            if ret and test_frame is not None:
                log.info("Successfully acquired stream from Pi Camera Ribbon Cable!")
                return adapter, preferred
            adapter.release()
        except Exception as exc:
            log.info("Picamera2 initialization info: %s", exc)

    # 2. Directly open standard VideoCapture for libcamerify / V4L2
    log.info("Opening camera device index %d ...", preferred)
    
    # Try V4L2 backend first, then default
    for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
        try:
            cap = cv2.VideoCapture(preferred, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Warmup loop: give camera up to 2 seconds to supply first valid frame
                for _ in range(20):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        log.info("Successfully acquired video stream on camera index %d!", preferred)
                        return cap, preferred
                    time.sleep(0.1)

                # If warmup failed for this backend, release before trying next
                log.warning("Warmup read timed out for backend %s on index %d. Releasing handle...", backend, preferred)
                cap.release()
                time.sleep(0.2)
        except Exception as exc:
            log.debug("VideoCapture error for backend %s: %s", backend, exc)

    log.warning("Opening fallback simple camera handle cap(%d) ...", preferred)
    cap = cv2.VideoCapture(preferred)
    return cap, preferred


def run_recognition_loop(client: Client) -> None:
    """
    Continuously capture and display camera frames at full 30 FPS,
    while face detection and recognition run asynchronously in a dedicated worker thread.
    """
    global last_decision_time, last_event_text, last_event_color
    global face_in_frame_status, cam_ready_status, latest_frame
    global current_status_label, current_status_color, pi_mode
    global active_face_locations, active_face_label, active_box_color, face_worker_running

    if face_recognition is None:
        log.error(
            "face_recognition library is not installed in this Python environment.\n"
            "Run on the Pi: pip install face-recognition"
        )
        sys.exit(1)

    # Initialise Pygame Screen
    pg_screen = None
    fonts = {}
    if PYGAME_AVAILABLE:
        try:
            pygame.init()
            pygame.font.init()
            pg_screen = pygame.display.set_mode((1024, 600))
            pygame.display.set_caption("Laboratory Access Control — Security Kiosk")

            fonts = {
                "title": pygame.font.SysFont("arial", 22, bold=True),
                "bold": pygame.font.SysFont("arial", 18, bold=True),
                "sub": pygame.font.SysFont("arial", 17, bold=True),
                "small": pygame.font.SysFont("arial", 14),
                "small_bold": pygame.font.SysFont("arial", 14, bold=True),
            }
            log.info("Pygame Kiosk UI Engine initialised (1024x600).")
        except Exception as exc:
            log.warning("Pygame GUI init failed: %s — falling back to OpenCV.", exc)
            pg_screen = None

    log.info("Initialising camera stream ...")
    cap, active_idx = smart_open_camera()

    # --- Background Face Recognition & Liveness Worker Thread ---
    def _new_session() -> ChallengeSession:
        return ChallengeSession(
            num_challenges=RECOGNITION_CHALLENGES,
            directions=(LIVENESS_LEFT, LIVENESS_RIGHT),
            require_recenter=False,
            timeout_s=12.0,
        )

    def _face_worker() -> None:
        global active_face_locations, active_face_label, active_box_color
        global face_in_frame_status, last_detected_name, last_decision_time
        log.info("⚡ Face recognition + challenge liveness worker started.")

        session = _new_session()
        tracked_full_box = None
        frame_counter = 0
        last_encoding_time = 0.0
        track_embs: list[tuple[float, np.ndarray]] = []   # (timestamp, embedding) for the current face track
        passive_scores: list[float] = []
        last_seen_time = 0.0
        spoof_hold_until = 0.0
        spoof_label = ""
        last_spoof_log_time = 0.0
        last_sched_denied_id = None
        last_sched_denied_time = 0.0
        last_processed_faculty_id = None
        last_processed_decision = None
        last_processed_time = 0.0
        no_face_frames_counter = 0

        def reset_track() -> None:
            nonlocal session, tracked_full_box, track_embs, passive_scores
            session = _new_session()
            tracked_full_box = None
            track_embs = []
            passive_scores = []

        def identify() -> tuple[str | None, str | None, float]:
            if not track_embs:
                return None, None, float("inf")
            recent = [e for _, e in track_embs[-FACE_MATCH_VOTES:]]
            return find_best_match(_l2_norm(np.mean(np.stack(recent, axis=0), axis=0)))

        while face_worker_running:
            try:
                if enrollment_capture_active:
                    reset_track()
                    time.sleep(0.05)
                    continue

                with pi_mode_lock:
                    current_mode = pi_mode

                if current_mode == "idle" or latest_frame is None:
                    active_face_locations = []
                    active_face_label = "SYSTEM IDLE - Maintenance Mode" if current_mode == "idle" else ""
                    active_box_color = (128, 128, 128)
                    reset_track()
                    time.sleep(0.1)
                    continue

                now = time.time()
                frame_counter += 1
                frame_copy = latest_frame.copy()
                rgb_full = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

                full_box, best_small = _detect_face_box_fast(
                    frame_copy,
                    prev_full_box=tracked_full_box,
                    use_roi_hint=(tracked_full_box is not None and frame_counter % 4 != 0),
                )

                face_in_frame_status = full_box is not None
                if full_box is None or best_small is None:
                    active_face_locations = []
                    active_face_label = ""
                    active_box_color = (255, 255, 255)
                    last_detected_name = ""
                    tracked_full_box = None
                    no_face_frames_counter += 1
                    if no_face_frames_counter > 20:
                        last_processed_faculty_id = None
                        last_processed_decision = None
                    # brief detector dropouts (e.g. mid head-turn) must not restart the challenge
                    if last_seen_time and (now - last_seen_time) > 1.2:
                        reset_track()
                    time.sleep(0.05)
                    continue

                no_face_frames_counter = 0
                # A big jump of the box = a different face/photo was swapped in -> new track
                if tracked_full_box is not None:
                    pt_, pr_, pb_, pl_ = tracked_full_box
                    t_, r_, b_, l_ = full_box
                    jump = np.hypot((l_ + r_ - pl_ - pr_) / 2.0, (t_ + b_ - pt_ - pb_) / 2.0)
                    if jump > 0.6 * max(1, pr_ - pl_):
                        reset_track()
                tracked_full_box = full_box
                last_seen_time = now
                active_face_locations = [best_small]

                usable, why = _face_box_usable(full_box, frame_copy.shape, MIN_FACE_SIZE_PX)
                if not usable:
                    active_box_color = (0, 165, 255)
                    active_face_label = why
                    last_detected_name = why
                    time.sleep(0.05)
                    continue

                if now < spoof_hold_until:
                    active_box_color = (0, 0, 255)
                    active_face_label = spoof_label
                    last_detected_name = "Spoof Attempt"
                    time.sleep(0.05)
                    continue

                with dlib_lock:
                    lm_list = face_recognition.face_landmarks(rgb_full, [full_box])
                pts = landmarks_to_array(lm_list[0]) if lm_list else None
                if pts is None:
                    time.sleep(0.03)
                    continue

                # Liveness runs in parallel with identification so the door decides faster
                live = session.update(pts, now=now) if REQUIRE_LIVENESS else None
                if PASSIVE_ANTISPOOF and minifasnet.is_loaded() and frame_counter % 2 == 0:
                    _, _, _, p_live = minifasnet.predict_liveness(frame_copy, full_box)
                    passive_scores.append(p_live)
                    passive_scores = passive_scores[-30:]

                # Identity: average of the last few embeddings on this face track
                if (now - last_encoding_time) >= 0.30:
                    with dlib_lock:
                        encs = face_recognition.face_encodings(rgb_full, [full_box], num_jitters=1, model="large")
                    if encs:
                        track_embs.append((now, _l2_norm(np.array(encs[0], dtype=np.float64))))
                        track_embs = track_embs[-6:]
                        last_encoding_time = now

                if len(track_embs) < 2:
                    active_box_color = (255, 200, 0)
                    active_face_label = "Scanning face..."
                    last_detected_name = "Scanning..."
                    time.sleep(0.02)
                    continue

                fid, fname, distance = identify()

                # Enrollment mode: preview only, the door never opens
                if current_mode == "enrollment":
                    active_box_color = (255, 200, 0)
                    active_face_label = "ENROLLMENT MODE - waiting for capture request from app"
                    time.sleep(0.04)
                    continue

                # --- UNKNOWN / UNREGISTERED FACE ---
                if fid is None:
                    if len(track_embs) >= FACE_MATCH_VOTES:
                        active_box_color = (0, 165, 255)
                        active_face_label = (
                            "Face Not Registered" if known_faces else "Face Not Found (No Registered Faces)"
                        )
                        last_detected_name = "Face Not Found"
                    else:
                        active_box_color = (255, 200, 0)
                        active_face_label = "Scanning face..."
                    time.sleep(0.04)
                    continue

                # --- SPOOF EVIDENCE ---
                passive_spoof, mean_live = _passive_spoof_verdict(passive_scores)
                if (live is not None and live.is_spoof) or passive_spoof:
                    reason = (live.prompt if (live is not None and live.is_spoof)
                              else f"Photo/screen detected (live score {mean_live:.2f})")
                    spoof_label = f"SPOOF BLOCKED: {reason}"
                    active_box_color = (0, 0, 255)
                    active_face_label = spoof_label
                    last_detected_name = "Spoof Attempt"
                    if (now - last_spoof_log_time) >= 6.0:
                        log.warning("🚫 SPOOF BLOCKED (claimed identity: %s): %s", fname, reason)
                        add_history_entry("Spoof Attempt", "denied", reason)
                        log_access(client, decision="denied", faculty_id=None,
                                   reason=f"Spoof blocked: {reason} (looked like {fname})")
                        last_spoof_log_time = now
                    spoof_hold_until = now + 3.0   # show red for 3 s, then start a fresh attempt
                    reset_track()
                    time.sleep(0.05)
                    continue

                # Debounce: same person still standing there after a decision
                if fid == last_processed_faculty_id and (now - last_processed_time) < 120.0:
                    if last_processed_decision == "granted":
                        active_box_color = (0, 255, 0)
                        active_face_label = f"WELCOME: {fname}"
                    else:
                        active_box_color = (0, 0, 255)
                        active_face_label = f"DENIED: {fname} (No Sched)"
                    last_detected_name = fname
                    time.sleep(0.08)
                    continue

                if (now - last_decision_time) < COOLDOWN_SECONDS:
                    active_box_color = (0, 255, 0)
                    active_face_label = f"{fname} — please wait"
                    time.sleep(0.06)
                    continue

                # --- LIVENESS CHALLENGE ---
                if live is not None and not live.is_live:
                    active_box_color = (0, 255, 200)
                    active_face_label = f"{fname}: {live.prompt}"
                    last_detected_name = f"{fname} (Verifying)"
                    time.sleep(0.02)
                    continue

                # Identity must still hold AFTER the challenge (no swap during the turn)
                if live is not None:
                    post = [e for t, e in track_embs if t >= session.passed_at]
                    if not post:
                        active_box_color = (0, 255, 180)
                        active_face_label = f"{fname}: Look straight at the camera"
                        time.sleep(0.02)
                        continue
                    pfid, _, _ = find_best_match(post[-1])
                    if pfid != fid:
                        log.warning("Identity changed during liveness challenge — restarting.")
                        reset_track()
                        continue

                faculty_id, name = fid, fname
                last_detected_name = name

                is_sched_ok, sched_msg, is_late, mins_late = check_faculty_schedule_status(faculty_id)
                if is_sched_ok:
                    unlock_door()
                    if is_late:
                        log.info("ACCESS GRANTED (LIVE, LATE %dm) - %s (%s, cos=%.4f)", mins_late, name, sched_msg, distance)
                        log_access(client, decision="granted", faculty_id=faculty_id,
                                   reason=f"Granted (Live Verified, Late by {mins_late}m): {name} | {sched_msg}")
                        active_box_color = (0, 165, 255)
                        active_face_label = f"GRANTED (LATE {mins_late}m): {name} [LIVE]"
                        add_history_entry(name, "late", f"Late by {mins_late}m ({sched_msg})")
                    else:
                        log.info("ACCESS GRANTED (LIVE, ON TIME) - %s (%s, cos=%.4f)", name, sched_msg, distance)
                        log_access(client, decision="granted", faculty_id=faculty_id,
                                   reason=f"Granted (Live Verified): {name} | {sched_msg}")
                        active_box_color = (0, 255, 0)
                        active_face_label = f"GRANTED: {name} [LIVE]"
                        add_history_entry(name, "granted", f"Schedule OK ({sched_msg})")
                    last_processed_decision = "granted"
                else:
                    active_box_color = (0, 0, 255)
                    active_face_label = f"DENIED: {name} (No Sched)"
                    if faculty_id != last_sched_denied_id or (now - last_sched_denied_time) >= 60.0:
                        log.info("ACCESS DENIED — %s: %s (cos=%.4f)", name, sched_msg, distance)
                        log_access(client, decision="denied", faculty_id=faculty_id, reason=f"Denied: {sched_msg}")
                        add_history_entry(name, "denied", sched_msg)
                        last_sched_denied_id = faculty_id
                        last_sched_denied_time = now
                    last_processed_decision = "denied"

                last_decision_time = now
                last_processed_faculty_id = faculty_id
                last_processed_time = now
                # Next attempt (or next person) needs a fresh challenge
                session = _new_session()
                time.sleep(0.08)
            except Exception as w_err:
                log.warning("Face worker error: %s", w_err)
                time.sleep(0.2)

    face_worker_running = True
    worker_t = threading.Thread(target=_face_worker, daemon=True, name="face-worker")
    worker_t.start()

    log.info("Recognition loop started — press Ctrl+C to stop.")
    window_title = "Lab Access Control — Security Kiosk"

    failed_frame_count = 0

    try:
        while True:
            # Handle Pygame quit events
            if PYGAME_AVAILABLE and pg_screen is not None:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_q):
                        return

            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    latest_frame = frame
                    cam_ready_status = True
                    failed_frame_count = 0
                else:
                    ret, frame = False, None
                    cam_ready_status = False
                    failed_frame_count += 1
            else:
                ret, frame = False, None
                cam_ready_status = False
                failed_frame_count += 1

            if not ret or frame is None or frame.size == 0:
                cam_ready_status = False
                if failed_frame_count >= 50 and failed_frame_count % 50 == 0:
                    log.warning("Camera stream offline. Attempting to re-acquire camera stream...")
                    try:
                        if cap is not None:
                            cap.release()
                    except Exception:
                        pass
                    cap, active_idx = smart_open_camera()

                if PYGAME_AVAILABLE and pg_screen is not None:
                    render_pygame_kiosk(pg_screen, fonts, camera_surface=None, active_idx=active_idx)
                else:
                    standby = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(standby, "SEARCHING FOR CAMERA STREAM...", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    if safe_show_frame(window_title, standby):
                        break
                time.sleep(0.05)
                continue

            # Convert BGR frame to Pygame Surface for Crisp UI rendering
            cam_surface = None
            if PYGAME_AVAILABLE and pg_screen is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_surface = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))

            # ------------------------------------------------------------------
            # Render UI immediately with latest asynchronous face data (30 FPS)
            # ------------------------------------------------------------------
            if PYGAME_AVAILABLE and pg_screen is not None:
                render_pygame_kiosk(
                    pg_screen,
                    fonts,
                    camera_surface=cam_surface,
                    face_locations=active_face_locations,
                    face_label=active_face_label,
                    box_color=active_box_color,
                    active_idx=active_idx,
                )
            else:
                frame_hud = draw_dashboard_overlay(
                    frame,
                    face_locations=active_face_locations,
                    face_label=active_face_label,
                    box_color=active_box_color,
                )
                if safe_show_frame(window_title, frame_hud):
                    break

            time.sleep(0.005)  # Yield GIL briefly so worker threads run smoothly

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received — shutting down.")
    finally:
        face_worker_running = False
        if cap is not None:
            cap.release()
        if PYGAME_AVAILABLE:
            try:
                pygame.quit()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        log.info("Camera released.")


# =============================================================================
#  ENTRY POINT - AUTO RUN EVERYTHING
# =============================================================================

def main() -> None:
    log.info("=" * 60)
    log.info("  Laboratory Access Control System — Door Guard v1.0")
    log.info("=" * 60)

    # 1. Initialise hardware
    setup_gpio()
    start_button_listener()

    # 2. Connect to Supabase
    client = init_supabase()

    # 3. Resolve which laboratory this Pi guards (by name or UUID)
    resolve_laboratory_id(client)

    # 4. Pre-load face registry and schedules from the database
    reload_known_faces(client)  # also calls reload_schedules() internally
    log.info("📅 Schedules loaded: %d entries for this laboratory.", len(known_schedules))
    
    # 5. Sync mode from database
    sync_pi_mode_with_database(client)
    log.info(f"📡 Initial Pi mode: {pi_mode.upper()}")

    # 6. Start background listener (handles enrollment, sync, status)
    start_realtime_listener(client)
    
    # 7. Start dedicated fast remote unlock listener (polls access_logs every 0.75s)
    start_remote_unlock_listener(client)

    # 8. Start HTTP control server (for app to change mode and enroll)
    start_enrollment_server(client, port=5000)

    # 9. Start the recognition loop (blocks until user quits with Ctrl+C)
    log.info("🚀 Door Guard running in AUTO mode. App controls everything.")
    log.info("📱 Use the app to switch between: recognition | enrollment | idle")
    run_recognition_loop(client)

    # 10. Clean up
    cleanup_gpio()
    log.info("Door Guard stopped. Goodbye.")


if __name__ == "__main__":
    main()