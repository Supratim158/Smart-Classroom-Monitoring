# backend/register_faces.py

import cv2
import os
import face_recognition
import pickle

def capture_and_save_face(name, face_dir='backend/face_data'):
    os.makedirs(face_dir, exist_ok=True)
    cam = cv2.VideoCapture(0)
    print("[INFO] Capturing face for:", name)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        cv2.imshow("Capture Face - Press 's' to save, 'q' to quit", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            image_path = os.path.join(face_dir, f"{name}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"[INFO] Saved: {image_path}")
            break
        elif key == ord('q'):
            print("[INFO] Capture cancelled.")
            break

    cam.release()
    cv2.destroyAllWindows()

def encode_faces(face_dir='backend/face_data'):
    known_encodings = []
    known_names = []

    for filename in os.listdir(face_dir):
        if filename.endswith(('.jpg', '.png')):
            name = os.path.splitext(filename)[0]
            path = os.path.join(face_dir, filename)
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)

            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(name)

    data = {"encodings": known_encodings, "names": known_names}
    with open("backend/encodings.pickle", "wb") as f:
        pickle.dump(data, f)
    print("[INFO] Face encodings saved.")
