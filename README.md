# Lab Access Control — Raspberry Pi Door Guard

Face-recognition based laboratory access control running on a Raspberry Pi 4.

## Hardware

| Component | Connection |
|---|---|
| Pi Camera Module | CSI ribbon cable |
| 12V Relay Module (IN) | GPIO 18 (BCM) |
| Solenoid Lock | Relay NO/COM terminals + 12V supply |
| Shared ground | Pi GND ↔ Relay GND ↔ 12V supply GND |

---

## Quick Start (on the Pi)

### 1. Transfer files
Copy this entire `facescan/` folder to your Pi. The easiest way is via SCP or a USB drive:
```bash
# From your Windows PC (PowerShell):
scp -r "C:\Users\Tom Pc\Desktop\Commissions\facescan" pi@<PI_IP>:/home/pi/
```

### 2. Run the setup script
```bash
cd /home/pi/facescan
chmod +x setup.sh
./setup.sh
```

The script will:
1. Set the timezone to **Asia/Manila**
2. Install all system-level build dependencies (`cmake`, `libopenblas`, etc.)
3. Enable the Pi Camera via `/boot/config.txt`
4. Create a Python virtual environment and install all packages
5. Prompt you to fill in your `.env` credentials
6. Install and enable a **systemd service** that starts the door guard automatically on boot

### 3. Fill in your `.env`
```bash
nano /home/pi/facescan/.env
```

| Variable | Value |
|---|---|
| `SUPABASE_URL` | `https://nbcoxrxywbrcjldccinr.supabase.co` |
| `SUPABASE_KEY` | Your Supabase **anon** or **service-role** key |
| `LABORATORY_NAME` | The lab name as stored in your `laboratories` table, e.g. `CCS Laboratory` |

### 4. Reboot
```bash
sudo reboot
```

The door guard starts automatically after boot. To check it:
```bash
sudo systemctl status door_guard
sudo journalctl -u door_guard -f   # live logs
```

---

## Manual Commands

```bash
# Start / stop the service
sudo systemctl start door_guard
sudo systemctl stop door_guard

# View live logs
sudo journalctl -u door_guard -f

# Run manually (for testing)
cd /home/pi/facescan
source venv/bin/activate
python door_guard.py
```

---

## How It Works

```
Camera → Downscale frame (50%) → Detect faces → Compute 128-d embedding
       → Compare against Supabase profiles → Euclidean distance < 0.6?
          ├── Outside operating hours (7 AM – 5 PM, Mon–Sat)?
          │     └── DENIED  → log_access("Outside operating hours")
          ├── Known face within hours?
          │     └── GRANTED → GPIO 18 LOW (5 s) → lock → log_access("granted")
          └── Unknown face?
                └── DENIED  → log_access("Unknown face")
```

### Overlay colours on the preview window
| Colour | Meaning |
|---|---|
| 🟢 Green | Access granted |
| 🔴 Red | Access denied — unknown face |
| 🟠 Orange | Access denied — outside operating hours |

---

## Realtime face enrolment

When you register a new face via the **React Native admin app**, the Pi's
Supabase Realtime subscription picks up the INSERT on the `profiles` table
and refreshes the in-memory face list automatically — no restart needed.

---

## File Reference

| File | Purpose |
|---|---|
| `door_guard.py` | Main application script |
| `requirements.txt` | Pinned Python dependencies |
| `.env.example` | Environment variable template |
| `.env` | Your actual credentials (never commit this) |
| `setup.sh` | One-shot installer for a fresh Pi OS |
| `door_guard.service` | systemd unit file (reference copy) |
