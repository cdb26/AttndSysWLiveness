import cv2
import os
import subprocess
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import sys
from datetime import datetime

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
CAM_W   = 520
CAM_H   = 390
MAX_SAMPLES = 150

# ─────────────────────────────────────────────────────────
#  GLOBALS
# ─────────────────────────────────────────────────────────
cap          = None
count        = 0
capturing    = False
dataset_path = ""

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ─────────────────────────────────────────────────────────
#  ROOT WINDOW
# ─────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Register New User — Face Attendance System")
root.geometry("900x660")
root.resizable(False, False)
root.configure(bg=BG)

# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────
def set_status(msg, color=ACCENT):
    status_var.set(msg)
    status_label.config(fg=color)

def set_progress(n):
    pct = int((n / MAX_SAMPLES) * 100)
    progress_var.set(f"{n} / {MAX_SAMPLES}  ({pct}%)")
    # draw bar
    bar_canvas.delete("all")
    bar_w = int((n / MAX_SAMPLES) * BAR_W)
    bar_canvas.create_rectangle(0, 0, BAR_W, BAR_H,
                                 fill=CARD, outline=DIMTEXT)
    if bar_w > 0:
        color = ACCENT if n < MAX_SAMPLES else YELLOW
        bar_canvas.create_rectangle(0, 0, bar_w, BAR_H,
                                     fill=color, outline="")

# ─────────────────────────────────────────────────────────
#  CAMERA FUNCTIONS
# ─────────────────────────────────────────────────────────
def start_camera():
    global cap

    name = name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a name first.")
        return
    if cap is not None:
        set_status("Camera already running", YELLOW)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        set_status("⚠  Could not open camera!", RED)
        cap = None
        return

    start_cam_btn.config(state="disabled", bg=DIMTEXT)
    set_status("● Camera live — press CAPTURE to start recording", ACCENT)
    update_frame()


def start_processing():
    global capturing, dataset_path, count

    name = name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a name first.")
        return
    if cap is None:
        messagebox.showerror("Error", "Start the camera first.")
        return

    dataset_path = os.path.join("dataset", name)
    os.makedirs(dataset_path, exist_ok=True)
    count     = 0
    capturing = True
    capture_btn.config(state="disabled", bg=DIMTEXT)
    set_status("⬤  Capturing face samples — hold still…", ACCENT)
    set_progress(0)


def update_frame():
    global cap, count, capturing

    if cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        root.after(30, update_frame)
        return

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if capturing and count < MAX_SAMPLES:
            count += 1
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{dataset_path}/{count}.jpg", face)
            set_progress(count)

        # draw box
        pct       = count / MAX_SAMPLES if capturing else 0
        box_color = (0, 229, 176) if capturing else (124, 92, 252)
        cv2.rectangle(frame, (x, y),    (x+w, y+h), box_color, 2)
        cv2.rectangle(frame, (x, y-30), (x+w, y),   box_color, -1)
        label_txt = f"Capturing {count}/{MAX_SAMPLES}" if capturing else "Face Detected"
        cv2.putText(frame, label_txt, (x+4, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (8, 12, 24), 2)

    # HUD
    fh, fw = frame.shape[:2]
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                (fw-110, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,229,176), 1)
    cv2.putText(frame, f"Faces: {len(faces)}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,229,176), 1)

    # Done!
    if count >= MAX_SAMPLES and capturing:
        capturing = False
        cv2.putText(frame, "CAPTURE COMPLETE", (fw//2-130, fh//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,229,176), 3)
        set_status("✔  Samples captured — training model…", YELLOW)
        push_frame(frame)
        root.after(1500, finish_registration)
        return

    push_frame(frame)
    root.after(16, update_frame)


def push_frame(frame):
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img   = Image.fromarray(rgb).resize((CAM_W, CAM_H), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img)
    cam_label.imgtk = imgtk
    cam_label.configure(image=imgtk)


def finish_registration():
    global cap

    if cap:
        cap.release()
        cap = None

    set_status("⚙  Running trainer — please wait…", YELLOW)
    root.update()

    result = subprocess.run([sys.executable, "train.py"],
                            capture_output=True, text=True)

    if result.returncode == 0:
        set_status("✔  Training complete! User registered successfully.", ACCENT)
        messagebox.showinfo("Done", "Training complete!\nUser has been registered.")
        subprocess.run([sys.executable, "main.py"])
        root.destroy()
    else:
        set_status("⚠  Training failed — check console.", RED)
        messagebox.showerror("Error", f"Training failed:\n{result.stderr}")

    root.destroy()


def on_close():
    global cap
    if cap:
        cap.release()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)


# ─────────────────────────────────────────────────────────
#  GUI LAYOUT
# ─────────────────────────────────────────────────────────

# TOP BAR
topbar = tk.Frame(root, bg=PANEL, height=56)
topbar.pack(fill="x")
topbar.pack_propagate(False)

tk.Label(topbar, text="◈  REGISTER NEW USER",
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

# ── LEFT: Camera ─────────────────────────────────────────
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
status_var   = tk.StringVar(value="■  Enter a name and start the camera")
status_label = tk.Label(left, textvariable=status_var,
                        font=("Courier", 10, "bold"),
                        bg=BG, fg=YELLOW, anchor="w")
status_label.pack(fill="x", pady=(6, 0))

# Progress bar
BAR_W = CAM_W
BAR_H = 14
bar_canvas = tk.Canvas(left, width=BAR_W, height=BAR_H,
                        bg=CARD, highlightthickness=0)
bar_canvas.pack(pady=2)
bar_canvas.create_rectangle(0, 0, BAR_W, BAR_H, fill=CARD, outline=DIMTEXT)

progress_var = tk.StringVar(value="0 / 150  (0%)")
tk.Label(left, textvariable=progress_var,
         font=("Courier", 9), bg=BG, fg=DIMTEXT).pack()

# Buttons
btn_row = tk.Frame(left, bg=BG)
btn_row.pack(pady=8)

start_cam_btn = tk.Button(btn_row,
    text="⬤  START CAMERA",
    font=("Courier", 11, "bold"),
    bg=VIOLET, fg="white", bd=0,
    activebackground="#5e48cc",
    padx=22, pady=10, cursor="hand2",
    command=start_camera)
start_cam_btn.pack(side="left", padx=8)

capture_btn = tk.Button(btn_row,
    text="◎  CAPTURE SAMPLES",
    font=("Courier", 11, "bold"),
    bg="#009970",
    fg="white", bd=0,
    activebackground=ACCENT,
    padx=22, pady=10, cursor="hand2",
    command=start_processing)
capture_btn.pack(side="left", padx=8)


# ── RIGHT: Info panel ────────────────────────────────────
right = tk.Frame(body, bg=PANEL, width=260)
right.pack(side="right", fill="y", padx=(12, 0))
right.pack_propagate(False)

tk.Label(right, text="USER INFO",
         font=("Courier", 11, "bold"),
         bg=PANEL, fg=VIOLET).pack(pady=(20, 4))
tk.Frame(right, bg=DIMTEXT, height=1).pack(fill="x", padx=10)

# Name input
name_frame = tk.Frame(right, bg=PANEL)
name_frame.pack(fill="x", padx=14, pady=16)

tk.Label(name_frame, text="FULL NAME",
         font=("Courier", 9), bg=PANEL, fg=DIMTEXT, anchor="w").pack(fill="x")

name_entry = tk.Entry(name_frame,
    font=("Courier", 13, "bold"),
    bg=CARD, fg=ACCENT,
    insertbackground=ACCENT,
    relief="flat", bd=0,
    highlightthickness=1,
    highlightbackground=DIMTEXT,
    highlightcolor=ACCENT)
name_entry.pack(fill="x", ipady=8, pady=4)

# Instructions card
tk.Frame(right, bg=DIMTEXT, height=1).pack(fill="x", padx=10, pady=(8,0))

instr_frame = tk.Frame(right, bg=CARD)
instr_frame.pack(fill="x", padx=10, pady=10)

tk.Label(instr_frame, text="HOW TO REGISTER",
         font=("Courier", 9, "bold"),
         bg=CARD, fg=ACCENT, anchor="w").pack(fill="x", padx=10, pady=(10,6))

steps = [
    ("1", "Enter the user's full name"),
    ("2", "Click START CAMERA"),
    ("3", "Click CAPTURE SAMPLES"),
    ("4", "Hold face steady in frame"),
    ("5", "Training runs automatically"),
]
for num, text in steps:
    row = tk.Frame(instr_frame, bg=CARD)
    row.pack(fill="x", padx=10, pady=2)
    tk.Label(row, text=num, font=("Courier", 9, "bold"),
             bg=ACCENT, fg=BG, width=2).pack(side="left")
    tk.Label(row, text=f"  {text}", font=("Courier", 9),
             bg=CARD, fg=TEXT, anchor="w").pack(side="left", fill="x")

tk.Frame(instr_frame, bg=CARD, height=8).pack()

# Liveness note
tk.Frame(right, bg=DIMTEXT, height=1).pack(fill="x", padx=10)

note = tk.Label(right,
    text="⚡  150 face samples are collected\nfor accurate recognition.\n\nBlink naturally — liveness\nchecking is active during\nattendance.",
    font=("Courier", 9),
    bg=PANEL, fg=DIMTEXT,
    justify="left", wraplength=220)
note.pack(padx=14, pady=14, anchor="w")

# BOTTOM ACCENT BAR
tk.Frame(root, bg=VIOLET, height=3).pack(fill="x", side="bottom")

root.mainloop()
