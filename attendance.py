import cv2
import numpy as np
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")
label_map = np.load("trainer/labels.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

blink_counter = 0
attendance_marked = set()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)

        eyes = eye_cascade.detectMultiScale(face)

        if len(eyes) == 0:
            blink_counter += 1
        else:
            if blink_counter > 2:
                name = label_map[label]

                if name not in attendance_marked:
                    with open("attendance.csv", "a") as f:
                        f.write(f"{name},{datetime.now()}\n")
                    attendance_marked.add(name)

                print(f"{name} marked present")

            blink_counter = 0

        cv2.putText(frame, label_map[label], (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()