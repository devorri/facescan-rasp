#!/usr/bin/env bash
# =============================================================================
#  Lab Access Control System — Raspberry Pi Setup Script
#  Run once on a fresh Raspberry Pi OS (Bullseye or Bookworm) installation.
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="door_guard"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
VENV_DIR="${SCRIPT_DIR}/venv"
PYTHON="python3"

echo ""
echo "============================================================"
echo "  Lab Access Control — Pi Setup"
echo "============================================================"
echo ""

# ── 1. Set timezone ──────────────────────────────────────────────────────────
echo "[1/7] Setting timezone to Asia/Manila ..."
sudo timedatectl set-timezone Asia/Manila
timedatectl

# ── 2. System dependencies ───────────────────────────────────────────────────
echo ""
echo "[2/7] Installing system dependencies ..."
sudo apt-get update -y
sudo apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    libjpeg-dev \
    libpng-dev

# ── 3. Enable Pi Camera (legacy camera stack for OpenCV) ─────────────────────
echo ""
echo "[3/7] Enabling legacy camera interface ..."
# Bookworm uses libcamera by default; OpenCV still works best with v4l2 driver.
# This adds 'start_x=1' and 'gpu_mem=128' to /boot/config.txt if not present.
if ! grep -q "^start_x=1" /boot/config.txt 2>/dev/null && \
   ! grep -q "^start_x=1" /boot/firmware/config.txt 2>/dev/null; then
    CONFIG_FILE="/boot/config.txt"
    [ -f "/boot/firmware/config.txt" ] && CONFIG_FILE="/boot/firmware/config.txt"
    echo "start_x=1" | sudo tee -a "$CONFIG_FILE"
    echo "gpu_mem=128" | sudo tee -a "$CONFIG_FILE"
    echo "  -> Camera config added to $CONFIG_FILE (reboot required after setup)."
else
    echo "  -> Camera already enabled in config.txt."
fi

# ── 4. Python virtual environment ────────────────────────────────────────────
echo ""
echo "[4/7] Creating Python virtual environment at ${VENV_DIR} ..."
"${PYTHON}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "  -> Upgrading pip ..."
pip install --upgrade pip

echo "  -> Installing Python packages (this may take 10-20 min for dlib) ..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

deactivate

# ── 5. Environment file ───────────────────────────────────────────────────────
echo ""
echo "[5/7] Checking environment file ..."
ENV_FILE="${SCRIPT_DIR}/.env"
if [ ! -f "${ENV_FILE}" ]; then
    cp "${SCRIPT_DIR}/.env.example" "${ENV_FILE}"
    echo ""
    echo "  !! .env file created from template."
    echo "  !! Open it now and fill in your SUPABASE_KEY and LABORATORY_NAME:"
    echo "       nano ${ENV_FILE}"
    echo ""
    read -r -p "  Press ENTER after you have saved your .env file to continue ..."
fi

# Validate required keys are set
source "${ENV_FILE}"
if [ -z "${SUPABASE_URL:-}" ] || [ -z "${SUPABASE_KEY:-}" ]; then
    echo "  ERROR: SUPABASE_URL and SUPABASE_KEY must be set in ${ENV_FILE}"
    exit 1
fi
if [ -z "${LABORATORY_NAME:-}" ] && [ -z "${LABORATORY_ID:-}" ]; then
    echo "  ERROR: Set either LABORATORY_NAME or LABORATORY_ID in ${ENV_FILE}"
    exit 1
fi
echo "  -> Environment variables look good."

# ── 6. systemd service ───────────────────────────────────────────────────────
echo ""
echo "[6/7] Installing systemd service (${SERVICE_NAME}) ..."

sudo tee "${SERVICE_FILE}" > /dev/null <<EOF
[Unit]
Description=Lab Access Control — Door Guard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${SCRIPT_DIR}
EnvironmentFile=${SCRIPT_DIR}/.env
ExecStart=${VENV_DIR}/bin/python ${SCRIPT_DIR}/door_guard.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"
echo "  -> Service installed and enabled on boot."

# ── 7. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Commands:"
echo "    Start now  : sudo systemctl start ${SERVICE_NAME}"
echo "    Stop       : sudo systemctl stop ${SERVICE_NAME}"
echo "    View logs  : sudo journalctl -u ${SERVICE_NAME} -f"
echo "    Status     : sudo systemctl status ${SERVICE_NAME}"
echo ""
echo "  NOTE: A REBOOT is required for the Pi Camera to activate."
echo "        Run: sudo reboot"
echo ""
