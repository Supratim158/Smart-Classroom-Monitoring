from flask import Flask, render_template, request
from backend.register_faces import capture_and_save_face, encode_faces
from backend.attendance import mark_attendance
from backend.detect_phone import detect_phone_and_log
import threading
import csv
import os

app = Flask(__name__)

# Home (dashboard)
@app.route('/')
def index():
    return render_template('index.html')

# Register Form
@app.route('/register')
def register_form():
    return render_template('register.html')

# Register Student (POST)
@app.route('/register', methods=['POST'])
def register_student():
    name = request.form.get('studentName')

    if not name:
        return "<h3 style='color:red;'>❌ Name missing!</h3><a href='/register'>Go back</a>"

    safe_name = name.strip().replace(" ", "_")

    def run_camera():
        capture_and_save_face(safe_name)
        encode_faces()

    threading.Thread(target=run_camera).start()

    return render_template('register_camera.html')

# Start Attendance (camera + face_recognition)
@app.route('/start-attendance', methods=['POST'])
def start_attendance():
    threading.Thread(target=mark_attendance).start()
    return render_template('taking_attendance.html')



# View Attendance Table
@app.route('/attendance-records')
def attendance_records():
    records = []
    csv_path = os.path.join("backend", "attendance.csv")  # ✅ Correct path

    if os.path.exists(csv_path):
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header
            for i, row in enumerate(reader, start=1):
                records.append([i] + row)  # Add Sr. No.

    return render_template('attendence.html', attendance=records)

@app.route('/start-webcam')
def start_webcam():
    def run_monitoring():
        detect_phone_and_log()  # opens webcam window with overlays

    threading.Thread(target=run_monitoring).start()

    return render_template('webcam_started.html')


@app.route('/alerts')
def view_alerts():
    alerts = []
    csv_path = os.path.join("backend", "alerts.csv")

    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                alerts.append(row)

    alerts = list(reversed(alerts))  # Reverse list in Python
    return render_template('alert.html', alerts=alerts)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
