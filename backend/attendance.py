# backend/attendance.py

import cv2
import face_recognition
import pickle
from datetime import datetime
import csv
import os

# Hardcoded name-to-roll mapping (can later be replaced with DB)
roll_map = {
    "Supratim_Modak": "101",
    "Rohit_Sharma": "102"
}

def mark_attendance():
    # Load encodings
    with open("backend/encodings.pickle", "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    cam = cv2.VideoCapture(0)
    print("[INFO] Starting webcam for attendance...")

    name_detected = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, face_locations)

        for encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"

            if True in matches:
                match_idx = matches.index(True)
                name = known_names[match_idx]
                name_detected = name
                break  # stop after first match

        if name_detected:
            break

        cv2.imshow("Attendance - Press 'q' to exit manually", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if name_detected:
        roll = roll_map.get(name_detected, "Unknown")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [name_detected.replace("_", " "), roll, "Present", timestamp]

        os.makedirs("backend", exist_ok=True)
        file_exists = os.path.isfile("backend/attendance.csv")

        with open("backend/attendance.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "RollNo", "Status", "Timestamp"])
            writer.writerow(row)

        print(f"[INFO] Attendance recorded for {name_detected}")
    else:
        print("[INFO] No known face detected.")
