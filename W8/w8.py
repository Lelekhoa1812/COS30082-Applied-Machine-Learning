import cv2
import face_recognition
import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog, messagebox
import time

# === Init global configs (directory setups) ===
known_faces_dir = 'known_faces'
snapshot_dir = 'snapshot'
os.makedirs(snapshot_dir, exist_ok=True)
os.makedirs(known_faces_dir, exist_ok=True)

# === Load known faces ===
known_face_encodings = []
known_face_names = []
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.png', '.JPG', '.PNG')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# === App state ===
unknown_faces = {}
saved_known_faces = set()
paused = False
current_frame = None

# === Save snapshot for known face (1 person each) ===
def save_snapshot(name, face_img):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    snapshot_file = os.path.join(snapshot_dir, f"{name}_{timestamp}.jpg")
    cv2.imwrite(snapshot_file, face_img)
    print(f"Saved snapshot: {snapshot_file}")

# === Handle prompt for unknown faces ===
def prompt_unknown_face(encoding, face_img):
    global paused
    paused = True
    # Submit the unknown
    def on_submit():
        person_name = name_input.get()
        if person_name:
            face_file = os.path.join(known_faces_dir, f"{person_name}.jpg")
            cv2.imwrite(face_file, face_img)
            known_face_encodings.append(encoding)
            known_face_names.append(person_name)
            saved_known_faces.add(person_name)
            print(f"Saved new identity: {person_name}")
        prompt_win.destroy()
        resume_stream()
    # Cancel changes
    def on_cancel():
        prompt_win.destroy()
        resume_stream()
    # Prompt new id 
    prompt_win = tk.Toplevel(root)
    prompt_win.title("Unknown Face Detected")
    tk.Label(prompt_win, text="Enter this person's name:").pack(pady=5)
    name_input = tk.Entry(prompt_win)
    name_input.pack(pady=5)
    tk.Button(prompt_win, text="OK", command=on_submit).pack(side='left', padx=10, pady=10)
    tk.Button(prompt_win, text="Cancel", command=on_cancel).pack(side='right', padx=10, pady=10)

# Continue streaming 
def resume_stream():
    global paused
    paused = False

# === Main frame update loop ===
def update_frame():
    global current_frame, paused
    if not paused:
        ret, frame = video.read()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frame = frame.copy()
        face_locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, encodings):
            name = "Unknown"
            distances = face_recognition.face_distance(known_face_encodings, encoding)
            if len(distances) > 0 and np.min(distances) < 0.5:
                match = np.argmin(distances)
                name = known_face_names[match]

            face_crop = frame[top:bottom, left:right]
            label = name
            box_color = (0, 255, 0)  # Green annotating recognition

            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), box_color, cv2.FILLED)
            cv2.putText(frame, label, (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if name != "Unknown" and name not in saved_known_faces:
                save_snapshot(name, face_crop)
                saved_known_faces.add(name)
            elif name == "Unknown":
                uid = str(hash(tuple(encoding.round(4))))
                if uid not in unknown_faces:
                    unknown_faces[uid] = {
                        "encoding": encoding,
                        "image": face_crop
                    }
                    prompt_unknown_face(encoding, face_crop)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk 
        video_label.configure(image=imgtk)

    root.after(16, update_frame)  # ~60 FPS (try to boost FPS but usually doesn't work)

# === Stop app handler ===
def stop_app():
    video.release()
    root.destroy()

# === GUI Setup (Tkinter) ===
root = tk.Tk()
root.title("Face Recognition")
video_label = tk.Label(root)
video_label.pack()
stop_btn = tk.Button(root, text="Stop", command=stop_app, bg='red', fg='white') # Stop button to terminate UI
stop_btn.pack(pady=10)

# === Start webcam ===
video = cv2.VideoCapture(0)
update_frame()
root.mainloop()
