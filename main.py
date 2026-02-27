import cv2
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import sys
import subprocess
import time
import threading

BG       = "#080c18"
PANEL    = "#0d1120"
CARD     = "#111827"
ACCENT   = "#00e5b0"
VIOLET   = "#7c5cfc"
RED      = "#ff4060"
YELLOW   = "#f0c040"
DIMTEXT  = "#3e5070"
TEXT     = "#ccd9f0"
CAM_W    = 560
CAM_H    = 420

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

CSV_FILE     = "attendance.csv"
DATASET_PATH = "dataset"
TRAINER_DIR  = "trainer"
MAX_SAMPLES  = 150

CONF_THRESHOLD = 70    
LOCK_FRAMES    = 10  
BLINK_MIN      = 3     
COOLDOWN_SEC   = 6.0  

recognizer   = cv2.face.LBPHFaceRecognizer_create()
TRAINER_YML  = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_NPY   = os.path.join(TRAINER_DIR, "labels.npy")


def load_model():
    global recognizer, label_map, model_loaded
    model_loaded = os.path.exists(TRAINER_YML) and os.path.exists(LABELS_NPY)
    label_map    = {}
    if model_loaded:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_YML)
        label_map = np.load(LABELS_NPY, allow_pickle=True).item()


load_model()

cap         = None
cam_running = False
current_tab = "attendance"  

att_mode          = None        
timed_in          = set()
timed_out         = set()
last_record_time  = {}

# Per-frame liveness vars
lock_count        = 0
locked_name       = None
eyes_was_closed   = False   
closed_frames     = 0      
blink_done        = False   

reg_capturing    = False
reg_count        = 0
reg_dataset_path = ""

root = tk.Tk()
root.title("Face Attendance System")
root.geometry("1060x720")
root.resizable(False, False)
root.configure(bg=BG)

def set_att_status(msg, color=ACCENT):
    att_status_var.set(msg)
    att_status_lbl.config(fg=color)


def set_reg_status(msg, color=ACCENT):
    reg_status_var.set(msg)
    reg_status_lbl.config(fg=color)


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

    tk.Label(row, text=name,   font=("Courier", 10, "bold"),
             bg=CARD, fg=color, width=13, anchor="w").pack(side="left")
    tk.Label(row, text=action, font=("Courier", 9),
             bg=CARD, fg=DIMTEXT, width=9).pack(side="left")
    tk.Label(row, text=time_str, font=("Courier", 9),
             bg=CARD, fg=DIMTEXT).pack(side="right", padx=8)

    log_canvas.update_idletasks()
    log_canvas.yview_moveto(1.0)


def record_attendance(name):
    now       = datetime.now()
    time_str  = now.strftime("%H:%M:%S")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    if time.time() - last_record_time.get(name, 0) < COOLDOWN_SEC:
        return

    last_record_time[name] = time.time()

    if att_mode == "IN":
        if name in timed_in:
            set_att_status(f"⚠  {name} already timed in", YELLOW)
            return
        timed_in.add(name)
        write_csv(name, "TIME IN", timestamp)
        add_log_row(name, "TIME IN", time_str)
        set_att_status(f"✔  {name} — TIME IN recorded", ACCENT)

    elif att_mode == "OUT":
        if name not in timed_in:
            set_att_status(f"⚠  {name} hasn't timed in yet!", RED)
            return
        if name in timed_out:
            set_att_status(f"⚠  {name} already timed out", YELLOW)
            return
        timed_out.add(name)
        write_csv(name, "TIME OUT", timestamp)
        add_log_row(name, "TIME OUT", time_str)
        set_att_status(f"✔  {name} — TIME OUT recorded", VIOLET)


def reset_liveness():
    global lock_count, locked_name, eyes_was_closed, closed_frames, blink_done
    lock_count      = 0
    locked_name     = None
    eyes_was_closed = False
    closed_frames   = 0
    blink_done      = False

def update_frame():
    global cap, cam_running
    global lock_count, locked_name, eyes_was_closed, closed_frames, blink_done
    global reg_capturing, reg_count

    if not cam_running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        root.after(30, update_frame)
        return

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    fh, fw = frame.shape[:2]

    now_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, now_str, (fw-110, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 176), 1)

    if current_tab == "attendance":
        mode_txt = f"MODE: {att_mode}" if att_mode else "SELECT A MODE"
        cv2.putText(frame, mode_txt, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 176), 1)

        if len(faces) == 0:
            reset_liveness()

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            name     = "Unknown"

            if model_loaded:
                lbl, conf = recognizer.predict(face_roi)
                if conf < CONF_THRESHOLD:
                    name = label_map.get(lbl, "Unknown")

            if name != "Unknown":
                if name == locked_name:
                    if lock_count < LOCK_FRAMES:
                        lock_count += 1
                else:
                    locked_name     = name
                    lock_count      = 1
                    eyes_was_closed = False
                    closed_frames   = 0
                    blink_done      = False
            else:
                reset_liveness()

            is_locked = (lock_count >= LOCK_FRAMES)

            if is_locked and not blink_done:
                eyes = eye_cascade.detectMultiScale(
                    face_roi,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(15, 15)
                )
                eyes_open_now = len(eyes) >= 1

                if not eyes_open_now:
                    closed_frames  += 1
                    eyes_was_closed = True
                else:
                    if eyes_was_closed and closed_frames >= BLINK_MIN:
                        blink_done = True
                        if att_mode is not None:
                            record_attendance(locked_name)
                    closed_frames   = 0
                    eyes_was_closed = False

            if name == "Unknown":
                box_col  = (255, 64, 96)
                disp_txt = "Unknown"
            elif not is_locked:
                box_col  = (240, 192, 64)
                disp_txt = f"{name}  locking {lock_count}/{LOCK_FRAMES}"
            elif blink_done:
                box_col  = (0, 229, 176)
                disp_txt = f"{name}  RECORDED ✔"
            else:
                box_col  = (0, 229, 176)
                disp_txt = f"{name}  BLINK NOW"

            cv2.rectangle(frame, (x, y),    (x+w, y+h), box_col, 2)
            cv2.rectangle(frame, (x, y-32), (x+w, y),   box_col, -1)
            cv2.putText(frame, disp_txt, (x+4, y-9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (8, 12, 24), 2)

            if is_locked and not blink_done and eyes_was_closed:
                bpct  = min(closed_frames / BLINK_MIN, 1.0)
                bx1, by1 = x, y + h + 5
                bx2, by2 = x + w, y + h + 13
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 30, 50), -1)
                cv2.rectangle(frame, (bx1, by1),
                              (bx1 + int((bx2-bx1)*bpct), by2),
                              (0, 229, 176), -1)

        cv2.putText(frame, f"LOCK {min(lock_count,LOCK_FRAMES)}/{LOCK_FRAMES}",
                    (8, fh-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (62, 80, 112), 1)

    elif current_tab == "register":
        cv2.putText(frame, "REGISTER MODE", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (124, 92, 252), 1)

        for (x, y, w, h) in faces:
            if reg_capturing and reg_count < MAX_SAMPLES:
                reg_count += 1
                face = gray[y:y+h, x:x+w]
                cv2.imwrite(f"{reg_dataset_path}/{reg_count}.jpg", face)
                # update progress bar on main thread
                root.after(0, lambda c=reg_count: update_reg_progress(c))

            pct       = reg_count / MAX_SAMPLES if reg_capturing else 0
            box_color = (124, 92, 252)
            cv2.rectangle(frame, (x, y),    (x+w, y+h), box_color, 2)
            cv2.rectangle(frame, (x, y-32), (x+w, y),   box_color, -1)
            lbl_txt = f"Capturing {reg_count}/{MAX_SAMPLES}" if reg_capturing else "Face Detected"
            cv2.putText(frame, lbl_txt, (x+4, y-9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (8, 12, 24), 2)

            # Sample fill bar
            if reg_capturing:
                bx1, by1 = x, y + h + 5
                bx2, by2 = x + w, y + h + 13
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 30, 50), -1)
                cv2.rectangle(frame, (bx1, by1),
                              (bx1 + int((bx2-bx1)*pct), by2),
                              (124, 92, 252), -1)

        if reg_capturing and reg_count >= MAX_SAMPLES:
            reg_capturing = False
            cv2.putText(frame, "CAPTURE DONE!", (fw//2-120, fh//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 229, 176), 3)
            push_frame(frame, "register")
            root.after(800, run_training)
            return

    push_frame(frame, current_tab)
    root.after(16, update_frame)


def push_frame(frame, tab):
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img   = Image.fromarray(rgb).resize((CAM_W, CAM_H), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img)
    if tab == "attendance":
        att_cam_label.imgtk = imgtk
        att_cam_label.configure(image=imgtk)
    else:
        reg_cam_label.imgtk = imgtk
        reg_cam_label.configure(image=imgtk)


def start_camera():
    global cap, cam_running
    if cam_running:
        return
    cap         = cv2.VideoCapture(0)
    cam_running = True
    update_frame()


def stop_camera():
    global cap, cam_running
    cam_running = False
    if cap:
        cap.release()
        cap = None
    att_cam_label.configure(image="")
    att_cam_label.config(text="[ Camera Off ]", font=("Courier", 12), fg=DIMTEXT)
    reg_cam_label.configure(image="")
    reg_cam_label.config(text="[ Camera Off ]", font=("Courier", 12), fg=DIMTEXT)

def set_att_mode(new_mode):
    global att_mode
    att_mode = new_mode
    reset_liveness()
    if new_mode == "IN":
        att_mode_lbl.config(text="▶  TIME IN MODE — look at camera, then blink", fg=ACCENT)
        btn_timein.config(bg="#003d2e", relief="sunken")
        btn_timeout.config(bg=CARD,    relief="flat")
    else:
        att_mode_lbl.config(text="◀  TIME OUT MODE — look at camera, then blink", fg=VIOLET)
        btn_timein.config(bg=CARD,      relief="flat")
        btn_timeout.config(bg="#1a0d3d", relief="sunken")
    set_att_status(f"Mode: {new_mode}  |  Face camera steadily → wait for lock → blink", TEXT)


def att_start():
    if not model_loaded:
        set_att_status("⚠  No trained model — register a user first!", RED)
        return
    start_camera()
    att_start_btn.config(state="disabled", bg=DIMTEXT)
    att_stop_btn.config(state="normal",   bg=RED)
    set_att_status("● LIVE — scanning faces…", ACCENT)


def att_stop():
    stop_camera()
    att_start_btn.config(state="normal",   bg=VIOLET)
    att_stop_btn.config(state="disabled", bg=DIMTEXT)
    set_att_status("■  Camera stopped", YELLOW)

def reg_start_camera():
    name = reg_name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Enter a name first.")
        return
    start_camera()
    reg_cam_btn.config(state="disabled", bg=DIMTEXT)
    set_reg_status("● Camera live — click CAPTURE to begin", ACCENT)


def reg_start_capture():
    global reg_capturing, reg_count, reg_dataset_path
    name = reg_name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Enter a name first.")
        return
    if not cam_running:
        messagebox.showerror("Error", "Start the camera first.")
        return

    reg_dataset_path = os.path.join(DATASET_PATH, name)
    os.makedirs(reg_dataset_path, exist_ok=True)
    reg_count     = 0
    reg_capturing = True
    reg_cap_btn.config(state="disabled", bg=DIMTEXT)
    update_reg_progress(0)
    set_reg_status("⬤  Capturing samples — keep face in frame…", ACCENT)


def update_reg_progress(n):
    pct = int(n / MAX_SAMPLES * 100)
    reg_progress_var.set(f"{n} / {MAX_SAMPLES}  ({pct}%)")
    bar_fill = int(n / MAX_SAMPLES * REG_BAR_W)
    reg_bar_canvas.delete("bar")
    if bar_fill > 0:
        col = ACCENT if n < MAX_SAMPLES else YELLOW
        reg_bar_canvas.create_rectangle(0, 0, bar_fill, REG_BAR_H,
                                        fill=col, outline="", tags="bar")


def run_training():
    set_reg_status("⚙  Training model — please wait…", YELLOW)
    root.update()
    stop_camera()

    def _train():
        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True, text=True
        )
        root.after(0, lambda: _on_train_done(result))

    threading.Thread(target=_train, daemon=True).start()


def _on_train_done(result):
    load_model() 
    if result.returncode == 0:
        set_reg_status("✔  Training complete! User registered.", ACCENT)
        messagebox.showinfo("Done", "Training complete!\nUser registered successfully.")
    else:
        set_reg_status("⚠  Training failed — check train.py", RED)
        messagebox.showerror("Error", f"Training failed:\n{result.stderr}")

    reg_name_entry.delete(0, tk.END)
    reg_cam_btn.config(state="normal", bg=VIOLET)
    reg_cap_btn.config(state="normal", bg="#009970")
    update_reg_progress(0)


# ═══════════════════════════════════════════════════════
#  TAB SWITCHING
# ═══════════════════════════════════════════════════════
def show_tab(tab):
    global current_tab
    current_tab = tab
    if tab == "attendance":
        att_frame.pack(fill="both", expand=True)
        reg_frame.pack_forget()
        tab_att_btn.config(bg=ACCENT, fg=BG)
        tab_reg_btn.config(bg=CARD,   fg=DIMTEXT)
    else:
        reg_frame.pack(fill="both", expand=True)
        att_frame.pack_forget()
        tab_att_btn.config(bg=CARD,   fg=DIMTEXT)
        tab_reg_btn.config(bg=VIOLET, fg="white")


def on_close():
    stop_camera()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)

# ═══════════════════════════════════════════════════════
#  ── TOP BAR ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════
#  ── TAB BAR ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════
tabbar = tk.Frame(root, bg="#060910", height=46)
tabbar.pack(fill="x")
tabbar.pack_propagate(False)

tab_att_btn = tk.Button(tabbar,
    text="  ◉  ATTENDANCE  ",
    font=("Courier", 11, "bold"),
    bg=ACCENT, fg=BG, bd=0, relief="flat",
    activebackground=ACCENT, activeforeground=BG,
    padx=10, pady=10, cursor="hand2",
    command=lambda: show_tab("attendance"))
tab_att_btn.pack(side="left", padx=(16, 4), pady=6)

tab_reg_btn = tk.Button(tabbar,
    text="  ✚  REGISTER USER  ",
    font=("Courier", 11, "bold"),
    bg=CARD, fg=DIMTEXT, bd=0, relief="flat",
    activebackground=VIOLET, activeforeground="white",
    padx=10, pady=10, cursor="hand2",
    command=lambda: show_tab("register"))
tab_reg_btn.pack(side="left", padx=4, pady=6)

# ═══════════════════════════════════════════════════════
#  ── CONTENT AREA ────────────────────────────────────────
# ═══════════════════════════════════════════════════════
content = tk.Frame(root, bg=BG)
content.pack(fill="both", expand=True)

# ───────────────────────────────────────────────────────
#  ATTENDANCE FRAME
# ───────────────────────────────────────────────────────
att_frame = tk.Frame(content, bg=BG)
# (packed by show_tab)

att_body = tk.Frame(att_frame, bg=BG)
att_body.pack(fill="both", expand=True, padx=14, pady=10)

# Left — camera
att_left = tk.Frame(att_body, bg=BG)
att_left.pack(side="left", fill="both")

att_cam_border = tk.Frame(att_left, bg=DIMTEXT, bd=1)
att_cam_border.pack()

att_cam_bg = tk.Frame(att_cam_border, bg="#05080f", width=CAM_W, height=CAM_H)
att_cam_bg.pack_propagate(False)
att_cam_bg.pack()

att_cam_label = tk.Label(att_cam_bg, bg="#05080f",
                          text="[ Camera Off ]",
                          font=("Courier", 12), fg=DIMTEXT)
att_cam_label.place(relx=0.5, rely=0.5, anchor="center")

att_status_var = tk.StringVar(value="■  Select a mode then start the camera")
att_status_lbl = tk.Label(att_left, textvariable=att_status_var,
                           font=("Courier", 10, "bold"),
                           bg=BG, fg=YELLOW, anchor="w")
att_status_lbl.pack(fill="x", pady=(5, 0))

att_mode_lbl = tk.Label(att_left, text="—  NO MODE SELECTED",
                         font=("Courier", 11, "bold"),
                         bg=PANEL, fg=DIMTEXT, pady=7, anchor="center")
att_mode_lbl.pack(fill="x", pady=4)

att_mode_row = tk.Frame(att_left, bg=BG)
att_mode_row.pack(pady=2)

btn_timein = tk.Button(att_mode_row,
    text="▶  TIME IN",
    font=("Courier", 11, "bold"),
    bg=CARD, fg=ACCENT, bd=1, relief="flat",
    activebackground="#003d2e",
    padx=24, pady=9, cursor="hand2",
    command=lambda: set_att_mode("IN"))
btn_timein.pack(side="left", padx=6)

btn_timeout = tk.Button(att_mode_row,
    text="◀  TIME OUT",
    font=("Courier", 11, "bold"),
    bg=CARD, fg=VIOLET, bd=1, relief="flat",
    activebackground="#1a0d3d",
    padx=24, pady=9, cursor="hand2",
    command=lambda: set_att_mode("OUT"))
btn_timeout.pack(side="left", padx=6)

att_ctrl_row = tk.Frame(att_left, bg=BG)
att_ctrl_row.pack(pady=4)

att_start_btn = tk.Button(att_ctrl_row,
    text="⬤  START CAMERA",
    font=("Courier", 11, "bold"),
    bg=VIOLET, fg="white", bd=0,
    activebackground="#5e48cc",
    padx=20, pady=9, cursor="hand2",
    command=att_start)
att_start_btn.pack(side="left", padx=6)

att_stop_btn = tk.Button(att_ctrl_row,
    text="■  STOP",
    font=("Courier", 11, "bold"),
    bg=DIMTEXT, fg="white", bd=0,
    activebackground=RED,
    padx=20, pady=9, cursor="hand2",
    state="disabled",
    command=att_stop)
att_stop_btn.pack(side="left", padx=6)

# Right — log panel
att_right = tk.Frame(att_body, bg=PANEL, width=280)
att_right.pack(side="right", fill="y", padx=(12, 0))
att_right.pack_propagate(False)

tk.Label(att_right, text="ATTENDANCE LOG",
         font=("Courier", 11, "bold"),
         bg=PANEL, fg=VIOLET).pack(pady=(12, 4))
tk.Frame(att_right, bg=DIMTEXT, height=1).pack(fill="x", padx=10)

log_canvas = tk.Canvas(att_right, bg=CARD, highlightthickness=0)
vsb = tk.Scrollbar(att_right, orient="vertical", command=log_canvas.yview)
log_canvas.configure(yscrollcommand=vsb.set)
vsb.pack(side="right", fill="y", padx=(0, 4), pady=6)
log_canvas.pack(fill="both", expand=True, padx=6, pady=6)

log_inner = tk.Frame(log_canvas, bg=CARD)
log_win   = log_canvas.create_window((0, 0), window=log_inner, anchor="nw")

log_inner.bind("<Configure>",
               lambda e: log_canvas.configure(scrollregion=log_canvas.bbox("all")))
log_canvas.bind("<Configure>",
                lambda e: log_canvas.itemconfig(log_win, width=e.width))

tk.Frame(att_right, bg=DIMTEXT, height=1).pack(fill="x", padx=10)
stats_row = tk.Frame(att_right, bg=PANEL)
stats_row.pack(fill="x", padx=10, pady=8)

c1 = tk.Frame(stats_row, bg=PANEL)
c1.pack(side="left", expand=True)
tk.Label(c1, text="TIME IN", font=("Courier", 8), bg=PANEL, fg=DIMTEXT).pack()
timein_count_var = tk.StringVar(value="0")
tk.Label(c1, textvariable=timein_count_var,
         font=("Courier", 22, "bold"), bg=PANEL, fg=ACCENT).pack()

c2 = tk.Frame(stats_row, bg=PANEL)
c2.pack(side="right", expand=True)
tk.Label(c2, text="TIME OUT", font=("Courier", 8), bg=PANEL, fg=DIMTEXT).pack()
timeout_count_var = tk.StringVar(value="0")
tk.Label(c2, textvariable=timeout_count_var,
         font=("Courier", 22, "bold"), bg=PANEL, fg=VIOLET).pack()

tk.Frame(att_right, bg=DIMTEXT, height=1).pack(fill="x", padx=10)
tk.Label(att_right,
    text="HOW TO RECORD\n\n"
         "1. Pick TIME IN or TIME OUT\n"
         "2. Start camera\n"
         "3. Look steadily at camera\n"
         "4. Wait for 'BLINK NOW'\n"
         "5. Blink once to confirm",
    font=("Courier", 8),
    bg=PANEL, fg=DIMTEXT, justify="left").pack(padx=12, pady=10, anchor="w")

def refresh_counts():
    timein_count_var.set(str(len(timed_in)))
    timeout_count_var.set(str(len(timed_out)))
    root.after(1000, refresh_counts)
refresh_counts()

# ───────────────────────────────────────────────────────
#  REGISTER FRAME
# ───────────────────────────────────────────────────────
reg_frame = tk.Frame(content, bg=BG)
# (packed by show_tab)

reg_body = tk.Frame(reg_frame, bg=BG)
reg_body.pack(fill="both", expand=True, padx=14, pady=10)

# Left — camera
reg_left = tk.Frame(reg_body, bg=BG)
reg_left.pack(side="left", fill="both")

reg_cam_border = tk.Frame(reg_left, bg=DIMTEXT, bd=1)
reg_cam_border.pack()

reg_cam_bg = tk.Frame(reg_cam_border, bg="#05080f", width=CAM_W, height=CAM_H)
reg_cam_bg.pack_propagate(False)
reg_cam_bg.pack()

reg_cam_label = tk.Label(reg_cam_bg, bg="#05080f",
                          text="[ Camera Off ]",
                          font=("Courier", 12), fg=DIMTEXT)
reg_cam_label.place(relx=0.5, rely=0.5, anchor="center")

reg_status_var = tk.StringVar(value="■  Enter a name and start the camera")
reg_status_lbl = tk.Label(reg_left, textvariable=reg_status_var,
                           font=("Courier", 10, "bold"),
                           bg=BG, fg=YELLOW, anchor="w")
reg_status_lbl.pack(fill="x", pady=(5, 0))

REG_BAR_W = CAM_W
REG_BAR_H = 12
reg_bar_canvas = tk.Canvas(reg_left, width=REG_BAR_W, height=REG_BAR_H,
                            bg=CARD, highlightthickness=1,
                            highlightbackground=DIMTEXT)
reg_bar_canvas.pack(pady=2)

reg_progress_var = tk.StringVar(value="0 / 150  (0%)")
tk.Label(reg_left, textvariable=reg_progress_var,
         font=("Courier", 9), bg=BG, fg=DIMTEXT).pack()

reg_btn_row = tk.Frame(reg_left, bg=BG)
reg_btn_row.pack(pady=6)

reg_cam_btn = tk.Button(reg_btn_row,
    text="⬤  START CAMERA",
    font=("Courier", 11, "bold"),
    bg=VIOLET, fg="white", bd=0,
    activebackground="#5e48cc",
    padx=20, pady=9, cursor="hand2",
    command=reg_start_camera)
reg_cam_btn.pack(side="left", padx=6)

reg_cap_btn = tk.Button(reg_btn_row,
    text="◎  CAPTURE SAMPLES",
    font=("Courier", 11, "bold"),
    bg="#009970", fg="white", bd=0,
    activebackground=ACCENT,
    padx=20, pady=9, cursor="hand2",
    command=reg_start_capture)
reg_cap_btn.pack(side="left", padx=6)

# Right — info panel
reg_right = tk.Frame(reg_body, bg=PANEL, width=280)
reg_right.pack(side="right", fill="y", padx=(12, 0))
reg_right.pack_propagate(False)

tk.Label(reg_right, text="USER INFO",
         font=("Courier", 11, "bold"),
         bg=PANEL, fg=VIOLET).pack(pady=(20, 4))
tk.Frame(reg_right, bg=DIMTEXT, height=1).pack(fill="x", padx=10)

name_frame = tk.Frame(reg_right, bg=PANEL)
name_frame.pack(fill="x", padx=14, pady=14)

tk.Label(name_frame, text="FULL NAME",
         font=("Courier", 9), bg=PANEL, fg=DIMTEXT, anchor="w").pack(fill="x")

reg_name_entry = tk.Entry(name_frame,
    font=("Courier", 13, "bold"),
    bg=CARD, fg=ACCENT,
    insertbackground=ACCENT,
    relief="flat", bd=0,
    highlightthickness=1,
    highlightbackground=DIMTEXT,
    highlightcolor=ACCENT)
reg_name_entry.pack(fill="x", ipady=8, pady=4)

tk.Frame(reg_right, bg=DIMTEXT, height=1).pack(fill="x", padx=10)

instr = tk.Frame(reg_right, bg=CARD)
instr.pack(fill="x", padx=10, pady=10)

tk.Label(instr, text="HOW TO REGISTER",
         font=("Courier", 9, "bold"),
         bg=CARD, fg=ACCENT, anchor="w").pack(fill="x", padx=10, pady=(10, 6))

for num, txt in [
    ("1", "Enter full name above"),
    ("2", "Click START CAMERA"),
    ("3", "Click CAPTURE SAMPLES"),
    ("4", "Hold face steady"),
    ("5", "Training runs automatically"),
]:
    r = tk.Frame(instr, bg=CARD)
    r.pack(fill="x", padx=10, pady=2)
    tk.Label(r, text=num, font=("Courier", 9, "bold"),
             bg=ACCENT, fg=BG, width=2).pack(side="left")
    tk.Label(r, text=f"  {txt}", font=("Courier", 9),
             bg=CARD, fg=TEXT, anchor="w").pack(side="left")

tk.Frame(instr, bg=CARD, height=8).pack()

tk.Frame(reg_right, bg=DIMTEXT, height=1).pack(fill="x", padx=10)
tk.Label(reg_right,
    text="⚡  150 samples captured\nfor accurate recognition.\n\nAttendance uses blink\nliveness verification.",
    font=("Courier", 8),
    bg=PANEL, fg=DIMTEXT, justify="left").pack(padx=14, pady=12, anchor="w")

tk.Frame(root, bg=ACCENT, height=3).pack(fill="x", side="bottom")

show_tab("attendance")

root.mainloop()