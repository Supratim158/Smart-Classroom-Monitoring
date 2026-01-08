import cv2
import face_recognition
import pickle
from ultralytics import YOLO
from datetime import datetime
import csv
import os
import time
import math
import mediapipe as mp

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing_utils = mp.solutions.drawing_utils

# Eye landmarks
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# EAR calculation
def calculate_EAR(landmarks, eye_indices, w, h):
    def get_point(index):
        x = int(landmarks[index].x * w)
        y = int(landmarks[index].y * h)
        return x, y

    p1, p2 = get_point(eye_indices[1]), get_point(eye_indices[5])
    p3, p4 = get_point(eye_indices[2]), get_point(eye_indices[4])
    p5, p6 = get_point(eye_indices[0]), get_point(eye_indices[3])

    vertical_1 = math.dist(p1, p2)
    vertical_2 = math.dist(p3, p4)
    horizontal = math.dist(p5, p6)

    EAR = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return EAR

# Thresholds
EAR_THRESHOLD = 0.20
SLEEP_SECONDS = 5

def detect_phone_and_log():
    model = YOLO('yolov8m.pt')
    cam = cv2.VideoCapture(0)
    sleep_tracker = {}

    print("[INFO] Webcam started...")

    while True:
        # Load encodings dynamically
        try:
            with open("backend/encodings.pickle", "rb") as f:
                data = pickle.load(f)
        except:
            data = {"encodings": [], "names": []}

        ret, frame = cam.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # --- Face Recognition ---
        small_rgb = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_rgb)
        face_encodings = face_recognition.face_encodings(small_rgb, face_locations)

        recognized_faces = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            if True in matches:
                name = data["names"][matches.index(True)]
            # Scale to original image size
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            recognized_faces.append({"name": name, "center": (center_x, center_y)})

        # --- Face Mesh + EAR + Name Overlay ---
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get nose position
                nose_x = int(face_landmarks.landmark[1].x * w)
                nose_y = int(face_landmarks.landmark[1].y * h)

                matched_name = "Unknown"
                min_dist = float("inf")
                for rf in recognized_faces:
                    dist = math.hypot(nose_x - rf["center"][0], nose_y - rf["center"][1])
                    if dist < min_dist and dist < 100:
                        matched_name = rf["name"]
                        min_dist = dist

                # Draw name
                cv2.putText(image, matched_name, (nose_x, nose_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 0), 2)

                # EAR
                right_ear = calculate_EAR(face_landmarks.landmark, RIGHT_EYE, w, h)
                left_ear = calculate_EAR(face_landmarks.landmark, LEFT_EYE, w, h)
                avg_ear = (right_ear + left_ear) / 2.0

                # Sleep Detection
                if avg_ear < EAR_THRESHOLD:
                    if matched_name not in sleep_tracker or sleep_tracker[matched_name] is None:
                        sleep_tracker[matched_name] = time.time()
                    elif time.time() - sleep_tracker[matched_name] > SLEEP_SECONDS:
                        cv2.putText(image, f"😴 {matched_name} Sleeping!", (nose_x, nose_y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        print(f"[SLEEP] {matched_name} is sleeping!")
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open("backend/alerts.csv", "a", newline='') as f:
                            csv.writer(f).writerow([now, matched_name, "Sleeping Detected"])
                else:
                    sleep_tracker[matched_name] = None

                # Draw Mesh
                drawing_utils.draw_landmarks(
                    image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1)
                )

        # --- Phone Detection ---
        phones_detected = False
        yolo_results = model.predict(image, imgsz=640, verbose=False)
        for r in yolo_results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                if model.names[int(cls.item())] == "cell phone":
                    phones_detected = True

        if phones_detected:
            for rf in recognized_faces:
                if rf["name"] != "Unknown":
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("backend/alerts.csv", "a", newline='') as f:
                        csv.writer(f).writerow([now, rf["name"], "Phone Usage"])
                    print(f"[ALERT] {rf['name']} is using phone at {now}")

        # Show feed
        cv2.imshow("Monitoring", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
