import cv2
import dlib
import numpy as np
import tkinter as tk
from tkinter import messagebox
import os
import pickle

# Load models
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# File paths
ENCODING_FILE = "face_encodings.pkl"
VOTE_LOG_FILE = "vote_log.pkl"

# Load DBs
if os.path.exists(ENCODING_FILE):
    with open(ENCODING_FILE, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

if os.path.exists(VOTE_LOG_FILE):
    with open(VOTE_LOG_FILE, "rb") as f:
        vote_log = pickle.load(f)
else:
    vote_log = {}

# Get encoding from 10 images
def average_encoding(images):
    encodings = []
    for frame in images:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = shape_predictor(frame, face)
            enc = face_encoder.compute_face_descriptor(frame, shape)
            encodings.append(np.array(enc))
            break
    if encodings:
        return np.mean(encodings, axis=0)
    return None

# Match face with accuracy
def match_face(encoding):
    best_match = None
    min_distance = float("inf")
    for aadhaar, info in face_db.items():
        stored_enc = info["encoding"]
        dist = np.linalg.norm(stored_enc - encoding)
        if dist < min_distance:
            min_distance = dist
            best_match = aadhaar
    if min_distance < 0.6:
        return best_match, 1 - min_distance  # Return accuracy as similarity
    return None, None

# ---------------------------- GUI Functions ---------------------------- #

def open_register_window():
    reg_window = tk.Toplevel(app)
    reg_window.title("Register New Voter")
    reg_window.geometry("400x400")

    tk.Label(reg_window, text="Enter Name").pack()
    name_entry = tk.Entry(reg_window)
    name_entry.pack()

    tk.Label(reg_window, text="Enter Aadhaar Number").pack()
    aadhaar_entry = tk.Entry(reg_window)
    aadhaar_entry.pack()

    captured_images = []

    def capture_faces():
        cap = cv2.VideoCapture(0)
        count = 0
        captured_images.clear()
        messagebox.showinfo("Capture", "Capturing 10 photos. Press 'q' to start.")

        while count < 10:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Captured: {count}/10", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    captured_images.append(frame.copy())
                    count += 1
                    break
            cv2.putText(frame, "Press 'q' to capture", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow("Capture Face", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Esc to stop early
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Done", f"Captured {len(captured_images)} photos.")

    def encode_and_save():
        name = name_entry.get().strip()
        aadhaar = aadhaar_entry.get().strip()

        if not name or not aadhaar:
            messagebox.showerror("Error", "Enter all details.")
            return
        if aadhaar in face_db:
            messagebox.showerror("Error", "Voter already registered.")
            return
        if len(captured_images) < 5:
            messagebox.showwarning("Warning", "Please capture at least 5 images.")
            return

        encoding = average_encoding(captured_images)
        if encoding is not None:
            face_db[aadhaar] = {"name": name, "encoding": encoding}
            with open(ENCODING_FILE, "wb") as f:
                pickle.dump(face_db, f)
            messagebox.showinfo("Success", "Voter registered.")
            reg_window.destroy()
        else:
            messagebox.showerror("Error", "Encoding failed.")

    tk.Button(reg_window, text="Capture Face (10 images)", command=capture_faces).pack(pady=10)
    tk.Button(reg_window, text="Encode & Register", command=encode_and_save, bg="green", fg="white").pack(pady=10)

def start_voting():
    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Voting", "Press 'q' to capture your face")
    encoding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Voting - Face Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                shape = shape_predictor(frame, face)
                encoding = np.array(face_encoder.compute_face_descriptor(frame, shape))
                break
            break

    cap.release()
    cv2.destroyAllWindows()

    if encoding is not None:
        aadhaar, accuracy = match_face(encoding)
        if aadhaar:
            if aadhaar in vote_log:
                messagebox.showinfo("Info", "You have already voted.")
                return

            voter_name = face_db[aadhaar]['name']
            vote_window = tk.Toplevel(app)
            vote_window.title("Vote Now")
            tk.Label(vote_window, text=f"Welcome {voter_name}! Accuracy: {round(accuracy*100, 2)}%").pack(pady=10)

            def vote(party):
                vote_log[aadhaar] = party
                with open(VOTE_LOG_FILE, "wb") as f:
                    pickle.dump(vote_log, f)
                messagebox.showinfo("Done", f"You voted for {party}")
                vote_window.destroy()

            tk.Button(vote_window, text="BJP", command=lambda: vote("BJP"), bg="orange").pack(pady=5)
            tk.Button(vote_window, text="CONGRESS", command=lambda: vote("CONGRESS"), bg="green").pack(pady=5)
            tk.Button(vote_window, text="BRS", command=lambda: vote("BRS"), bg="purple").pack(pady=5)

        else:
            messagebox.showerror("Error", "Face not recognized.")
    else:
        messagebox.showerror("Error", "Face not detected.")

def show_results():
    result = {"BJP": 0, "CONGRESS": 0, "BRS": 0}
    for vote in vote_log.values():
        if vote in result:
            result[vote] += 1
    res = "\n".join([f"{k}: {v} votes" for k, v in result.items()])
    messagebox.showinfo("Voting Result", res)

def exit_app():
    app.destroy()

# ---------------------------- Main App GUI ---------------------------- #
app = tk.Tk()
app.title("Smart Voting System with Face Recognition")
app.geometry("400x300")

tk.Button(app, text="New Register Voter", command=open_register_window, bg="blue", fg="white").pack(pady=15)
tk.Button(app, text="Start Voting", command=start_voting, bg="darkgreen", fg="white").pack(pady=15)
tk.Button(app, text="Voting Result", command=show_results, bg="black", fg="white").pack(pady=15)
tk.Button(app, text="Exit", command=exit_app, bg="red", fg="white").pack(pady=15)

app.mainloop()
