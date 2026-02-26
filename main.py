import tkinter as tk
import subprocess
import sys

def open_register():
    subprocess.run([sys.executable, "register.py"])
    root.destroy()

def start_attendance():
    subprocess.run([sys.executable, "attendance.py"])
    root.destroy()

def exit_app():
    root.destroy()

root = tk.Tk()
root.title("Face Attendance System")
root.geometry("500x350")
root.resizable(False, False)
root.configure(bg="#1e1e2f")

title = tk.Label(
    root,
    text="Face Recognition Attendance",
    font=("Arial", 18, "bold"),
    bg="#1e1e2f",
    fg="white"
)
title.pack(pady=40)

btn_style = {
    "font": ("Arial", 12, "bold"),
    "width": 20,
    "height": 2,
    "bd": 0
}

register_btn = tk.Button(
    root,
    text="Register New User",
    bg="#4CAF50",
    fg="white",
    command=open_register,
    **btn_style
)
register_btn.pack(pady=15)

attendance_btn = tk.Button(
    root,
    text="Start Attendance",
    bg="#FF9800",
    fg="white",
    command=start_attendance,
    **btn_style
)
attendance_btn.pack(pady=15)

exit_btn = tk.Button(
    root,
    text="Exit",
    bg="#f44336",
    fg="white",
    command=exit_app,
    **btn_style
)
exit_btn.pack(pady=20)

root.mainloop()