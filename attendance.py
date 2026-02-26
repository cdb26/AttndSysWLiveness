import subprocess

import cv2
import numpy as np
from datetime import datetime
import tkinter as tk
from PIL import Image, ImageTk
import os
import sys
import time

# ─────────────────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────────────────
BG      = "#080c18"
PANEL   = "#0d1120"
CARD    = "#111827"
ACCENT  = "#00e5b0"
VIOLET  = "#7c5cfc"
RED     = "#ff4060"
YELLOW  = "#f0c040"
DIMTEXT = "#3e5070"
TEXT    = "#ccd9f0"
CAM_W   = 580
CAM_H   = 435

# ─────────────────────────────────────────────────────────
#  LIVENESS CONFIG
#
#  How it works:
#  1. Face must be recognized (confidence < CONF_THRESHOLD) for
#     LOCK_FRAMES consecutive frames  → "locked on"
#  2. Once locked, the system waits for ONE blink
#     (eyes disappear ≥ BLINK_MIN_FRAMES then reappear)
#  3. On blink confirmed → attendance recorded, 5-sec cooldown
# ─────────────────────────────────────────────────────────
CONF_THRESHOLD  = 70    # lower = stricter match
LOCK_FRAMES     = 8     # frames face must be stable before blink check
BLINK_MIN       = 2     # min consecutive no-eye frames to count as blink
RECORD_COOLDOWN = 5.0   # seconds before same person can record again

# ─────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────
recognizer   = cv2.face.LBPHFaceRecognizer_create()
TRAINER_YML  = "trainer/trainer.yml"
LABELS_NPY   = "trainer/labels.npy"
model_loaded = os.path.exists(TRAINER_YML) and os.path.exists(LABELS_NPY)
label_map    = {}

if model_loaded:
    recognizer.read(TRAINER_YML)
    label_map = np.load(LABELS_NPY, allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

CSV_FILE = "attendance.csv"

# ─────────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────────
cap           = None
running       = False
mode          = None          # "IN" or "OUT"
timed_in      = set()
timed_out     = set()

# Per-frame liveness tracking
lock_count    = 0             # consecutive frames with same recognized face
locked_name   = None          # name currently being tracked
eyes_closed_count = 0         # consecutive frames with no eyes (blink tracker)
blink_detected    = False     # blink confirmed flag
last_record_time  = {}        # name → timestamp of last record (cooldown)

# ─────────────────────────────────────────────────────────
#  ROOT WINDOW
# ─────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Attendance System — Face Recognition")
root.geometry("980x700")
root.resizable(False, False)
root.configure(bg=BG)

# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────
def set_status(msg, color=ACCENT):
    status_var.set(msg)
    status_label.config(fg=color)


def write_csv(name, action, timestamp):
    with open(CSV_FILE, "a") as f:
        f.write(f"{name},{action},{timestamp}\n")


def add_log_row(name, action, time_str):
    color = ACCENT if action == "TIME IN" else VIOLET
    row   = tk.Frame(log_inner, bg=CARD, pady=3)
    row.pack(fill="x", padx=4, pady=2)

    dot = tk.Canvas(row, width=8, height=8, bg=CARD, highlightthickness=0)
    dot.pack(side="left", padx=(8, 6))
    dot.create_oval(1, 1, 7, 7, fill=color, outline="")

    tk.Label(row, text=name, font=("Courier", 10, "bold"),
             bg=CARD, fg=color, width=14, anchor="w").pack(side="left")
    tk.Label(row, text=action, font=("Courier", 9),
             bg=CARD, fg=DIMTEXT, width=9).pack(side="left")
    tk.Label(row, text=time_str, font=("Courier", 9),
             bg=CARD, fg=DIMTEXT).pack(side="right", padx=8)

    log_canvas.update_idletasks()
    log_canvas.yview_moveto(1.0)


def record_attendance(name):
    """Write attendance record after liveness is confirmed."""
    now       = datetime.now()
    time_str  = now.strftime("%H:%M:%S")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Cooldown guard
    last = last_record_time.get(name, 0)
    if time.time() - last < RECORD_COOLDOWN:
        return

    last_record_time[name] = time.time()

    if mode == "IN":
        if name in timed_in:
            set_status(f"⚠  {name} already timed in", YELLOW)
            return
        timed_in.add(name)
        write_csv(name, "TIME IN", timestamp)
        add_log_row(name, "TIME IN", time_str)
        set_status(f"✔  {name} — TIME IN recorded", ACCENT)

    elif mode == "OUT":
        if name not in timed_in:
            set_status(f"⚠  {name} hasn't timed in yet!", RED)
            return
        if name in timed_out:
            set_status(f"⚠  {name} already timed out", YELLOW)
            return
        timed_out.add(name)
        write_csv(name, "TIME OUT", timestamp)
        add_log_row(name, "TIME OUT", time_str)
        set_status(f"✔  {name} — TIME OUT recorded", VIOLET)


# ─────────────────────────────────────────────────────────
#  CAMERA LOOP
# ─────────────────────────────────────────────────────────
def update_frame():
    global lock_count, locked_name, eyes_closed_count, blink_detected

    if not running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        root.after(30, update_frame)
        return

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    fh, fw = frame.shape[:2]

    if len(faces) == 0:
        # No face — reset lock
        lock_count         = 0
        locked_name        = None
        eyes_closed_count  = 0
        blink_detected     = False

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        name     = "Unknown"
        conf     = 999

        if model_loaded:
            lbl, conf = recognizer.predict(face_roi)
            if conf < CONF_THRESHOLD:
                name = label_map.get(lbl, "Unknown")

        # ── Phase 1: Lock onto face ──────────────────────
        if name != "Unknown":
            if name == locked_name:
                lock_count += 1
            else:
                locked_name       = name
                lock_count        = 1
                eyes_closed_count = 0
                blink_detected    = False
        else:
            lock_count  = 0
            locked_name = None

        # ── Phase 2: Blink detection once locked ─────────
        is_locked  = lock_count >= LOCK_FRAMES
        eyes       = eye_cascade.detectMultiScale(
                         face_roi,
                         scaleFactor=1.1,
                         minNeighbors=5,
                         minSize=(20, 20)) if is_locked else []

        if is_locked and not blink_detected:
            if len(eyes) == 0:
                eyes_closed_count += 1
            else:
                if eyes_closed_count >= BLINK_MIN:
                    # Blink confirmed!
                    blink_detected = True
                    if mode is not None:
                        record_attendance(locked_name)
                eyes_closed_count = 0

        # ── Draw overlay ─────────────────────────────────
        if name == "Unknown":
            box_col  = (255, 64, 96)
            disp_txt = "Unknown"
        else:
            box_col  = (0, 229, 176) if is_locked else (240, 192, 64)
            conf_txt = f"{100 - int(conf)}%"
            if not is_locked:
                disp_txt = f"{name} ({conf_txt}) locking..."
            elif blink_detected:
                disp_txt = f"{name} ✔ BLINK OK"
            else:
                disp_txt = f"{name} ({conf_txt}) — BLINK NOW"

        cv2.rectangle(frame, (x, y),    (x+w, y+h), box_col, 2)
        cv2.rectangle(frame, (x, y-34), (x+w, y),   box_col, -1)
        cv2.putText(frame, disp_txt, (x+4, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (8, 12, 24), 2)

        # Lock progress bar under face box
        if is_locked and not blink_detected:
            bar_x1, bar_y1 = x, y + h + 6
            bar_x2, bar_y2 = x + w, y + h + 14
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (30,40,60), -1)
            cv2.rectangle(frame, (bar_x1, bar_y1),
                          (bar_x1 + min(w, int(w * min(eyes_closed_count/BLINK_MIN,1))), bar_y2),
                          (0,229,176), -1)

        # Eye dots
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(face_roi, (ex + ew//2, ey + eh//2), 3, (0,229,176), -1)

    # ── HUD ──────────────────────────────────────────────
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                (fw-110, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,229,176), 1)
    mode_txt = f"MODE: {mode}" if mode else "MODE: NONE — select a mode"
    cv2.putText(frame, mode_txt, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,229,176), 1)

    lock_txt = f"LOCK: {min(lock_count, LOCK_FRAMES)}/{LOCK_FRAMES}"
    cv2.putText(frame, lock_txt, (8, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (62, 80, 112), 1)

    # ── Push to Tkinter ───────────────────────────────────
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img   = Image.fromarray(rgb).resize((CAM_W, CAM_H), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img)
    cam_label.imgtk = imgtk
    cam_label.configure(image=imgtk)

    root.after(16, update_frame)


def start_camera():
    global cap, running, lock_count, locked_name, eyes_closed_count, blink_detected

    if running:
        return
    if not model_loaded:
        set_status("⚠  No trained model — register users first!", RED)
        return

    cap               = cv2.VideoCapture(0)
    running           = True
    lock_count        = 0
    locked_name       = None
    eyes_closed_count = 0
    blink_detected    = False

    start_btn.config(state="disabled", bg=DIMTEXT)
    stop_btn.config(state="normal", bg=RED)
    set_status("● LIVE — Scanning faces…", ACCENT)
    update_frame()


def stop_camera():
    global cap, running

    running = False
    if cap:
        cap.release()
        cap = None
        subprocess.run([sys.executable, "main.py"])
        root.destroy()

    cam_label.configure(image="")
    cam_label.config(text="[ Camera Off ]", font=("Courier", 13), fg=DIMTEXT)
    set_status("■  Camera stopped", YELLOW)
    start_btn.config(state="normal", bg=VIOLET)
    stop_btn.config(state="disabled", bg=DIMTEXT)


def set_mode(new_mode):
    global mode
    mode = new_mode
    if new_mode == "IN":
        mode_indicator.config(text="▶  TIME IN MODE  —  look at camera, then blink", fg=ACCENT)
        timein_btn.config(bg="#003d2e", relief="sunken")
        timeout_btn.config(bg=CARD,    relief="flat")
    else:
        mode_indicator.config(text="◀  TIME OUT MODE  —  look at camera, then blink", fg=VIOLET)
        timein_btn.config(bg=CARD,      relief="flat")
        timeout_btn.config(bg="#1a0d3d", relief="sunken")
    set_status(f"Mode: {new_mode}  |  Face the camera steadily, then blink once", TEXT)


def on_close():
    stop_camera()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)

# ─────────────────────────────────────────────────────────
#  GUI LAYOUT
# ─────────────────────────────────────────────────────────

# TOP BAR
topbar = tk.Frame(root, bg=PANEL, height=56)
topbar.pack(fill="x")
topbar.pack_propagate(False)

tk.Label(topbar, text="◈  FACE ATTENDANCE SYSTEM",
         font=("Courier", 15, "bold"),
         bg=PANEL, fg=ACCENT).pack(side="left", padx=20, pady=14)

clock_var = tk.StringVar()
tk.Label(topbar, textvariable=clock_var,
         font=("Courier", 11), bg=PANEL, fg=DIMTEXT).pack(side="right", padx=20)

def tick():
    clock_var.set(datetime.now().strftime("%A  %d %b %Y   %H:%M:%S"))
    root.after(1000, tick)
tick()

# BODY
body = tk.Frame(root, bg=BG)
body.pack(fill="both", expand=True, padx=14, pady=10)

# LEFT
left = tk.Frame(body, bg=BG)
left.pack(side="left", fill="both")

cam_border = tk.Frame(left, bg=DIMTEXT, bd=1)
cam_border.pack()

cam_bg = tk.Frame(cam_border, bg="#05080f", width=CAM_W, height=CAM_H)
cam_bg.pack_propagate(False)
cam_bg.pack()

cam_label = tk.Label(cam_bg, bg="#05080f",
                     text="[ Camera Off ]",
                     font=("Courier", 13), fg=DIMTEXT)
cam_label.place(relx=0.5, rely=0.5, anchor="center")

# Status
status_var   = tk.StringVar(value="■  Idle — select a mode then start the camera")
status_label = tk.Label(left, textvariable=status_var,
                        font=("Courier", 10, "bold"),
                        bg=BG, fg=YELLOW, anchor="w")
status_label.pack(fill="x", pady=(6, 0))

# Mode indicator
mode_indicator = tk.Label(left,
    text="—  NO MODE SELECTED",
    font=("Courier", 11, "bold"),
    bg=PANEL, fg=DIMTEXT, pady=7, anchor="center")
mode_indicator.pack(fill="x", pady=4)

# Mode buttons
mode_row = tk.Frame(left, bg=BG)
mode_row.pack(pady=2)

timein_btn = tk.Button(mode_row,
    text="▶  TIME IN",
    font=("Courier", 11, "bold"),
    bg=CARD, fg=ACCENT, bd=1, relief="flat",
    activebackground="#003d2e",
    padx=26, pady=10, cursor="hand2",
    command=lambda: set_mode("IN"))
timein_btn.pack(side="left", padx=8)

timeout_btn = tk.Button(mode_row,
    text="◀  TIME OUT",
    font=("Courier", 11, "bold"),
    bg=CARD, fg=VIOLET, bd=1, relief="flat",
    activebackground="#1a0d3d",
    padx=26, pady=10, cursor="hand2",
    command=lambda: set_mode("OUT"))
timeout_btn.pack(side="left", padx=8)

# Camera controls
ctrl_row = tk.Frame(left, bg=BG)
ctrl_row.pack(pady=4)

start_btn = tk.Button(ctrl_row,
    text="⬤  START CAMERA",
    font=("Courier", 11, "bold"),
    bg=VIOLET, fg="white", bd=0,
    activebackground="#5e48cc",
    padx=22, pady=10, cursor="hand2",
    command=start_camera)
start_btn.pack(side="left", padx=8)

stop_btn = tk.Button(ctrl_row,
    text="■  STOP",
    font=("Courier", 11, "bold"),
    bg=DIMTEXT, fg="white", bd=0,
    activebackground=RED,
    padx=22, pady=10, cursor="hand2",
    state="disabled",
    command=stop_camera)
stop_btn.pack(side="left", padx=8)

# RIGHT PANEL
right = tk.Frame(body, bg=PANEL, width=290)
right.pack(side="right", fill="y", padx=(12, 0))
right.pack_propagate(False)

tk.Label(right, text="ATTENDANCE LOG",
         font=("Courier", 11, "bold"),
         bg=PANEL, fg=VIOLET).pack(pady=(14, 4))
tk.Frame(right, bg=DIMTEXT, height=1).pack(fill="x", padx=10)

log_canvas = tk.Canvas(right, bg=CARD, highlightthickness=0)
vsb = tk.Scrollbar(right, orient="vertical", command=log_canvas.yview)
log_canvas.configure(yscrollcommand=vsb.set)
vsb.pack(side="right", fill="y", padx=(0, 4), pady=6)
log_canvas.pack(fill="both", expand=True, padx=6, pady=6)

log_inner = tk.Frame(log_canvas, bg=CARD)
log_win   = log_canvas.create_window((0, 0), window=log_inner, anchor="nw")

def _on_inner_configure(e):
    log_canvas.configure(scrollregion=log_canvas.bbox("all"))

def _on_canvas_configure(e):
    log_canvas.itemconfig(log_win, width=e.width)

log_inner.bind("<Configure>", _on_inner_configure)
log_canvas.bind("<Configure>", _on_canvas_configure)

# Stats
tk.Frame(right, bg=DIMTEXT, height=1).pack(fill="x", padx=10)
stats = tk.Frame(right, bg=PANEL)
stats.pack(fill="x", padx=12, pady=10)

col1 = tk.Frame(stats, bg=PANEL)
col1.pack(side="left", expand=True)
tk.Label(col1, text="TIME IN", font=("Courier", 8),
         bg=PANEL, fg=DIMTEXT).pack()
present_count_var = tk.StringVar(value="0")
tk.Label(col1, textvariable=present_count_var,
         font=("Courier", 22, "bold"), bg=PANEL, fg=ACCENT).pack()

col2 = tk.Frame(stats, bg=PANEL)
col2.pack(side="right", expand=True)
tk.Label(col2, text="TIME OUT", font=("Courier", 8),
         bg=PANEL, fg=DIMTEXT).pack()
out_count_var = tk.StringVar(value="0")
tk.Label(col2, textvariable=out_count_var,
         font=("Courier", 22, "bold"), bg=PANEL, fg=VIOLET).pack()

# Hint label
tk.Frame(right, bg=DIMTEXT, height=1).pack(fill="x", padx=10, pady=(4,0))
tk.Label(right,
    text="HOW TO RECORD\n\n"
         "1. Select TIME IN or TIME OUT\n"
         "2. Start the camera\n"
         "3. Look steadily at camera\n"
         "4. Wait for 'BLINK NOW' prompt\n"
         "5. Blink once to confirm",
    font=("Courier", 8),
    bg=PANEL, fg=DIMTEXT,
    justify="left").pack(padx=14, pady=10, anchor="w")

def refresh_counts():
    present_count_var.set(str(len(timed_in)))
    out_count_var.set(str(len(timed_out)))
    root.after(1000, refresh_counts)
refresh_counts()

# Bottom accent
tk.Frame(root, bg=ACCENT, height=3).pack(fill="x", side="bottom")

root.mainloop()