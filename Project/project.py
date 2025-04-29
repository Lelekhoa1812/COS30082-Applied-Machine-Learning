import cv2
import face_recognition
from deepface import DeepFace # Weight are saved at upon importing Users/.../.deepface/weights/facial_expression_model_weights.h5
import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog, messagebox
import time
from skimage.feature import local_binary_pattern # Binary the image's features to identify liveness

# === Init global config ===
known_faces_dir = 'known_faces'
snapshot_dir = 'snapshot'
os.makedirs(snapshot_dir, exist_ok=True)
os.makedirs(known_faces_dir, exist_ok=True)

# === Load known faces ===
known_face_encodings = []
known_face_names = []
# Process and encode each of the images within known-set directory
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.png', '.JPG', '.PNG')): # Models only support these file formats (not .avif, .webp or .heic)
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings: # Encode image data into vectors and split name from filename
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# === App state ===
unknown_faces = {}
saved_known_faces = set()
paused = False # Set true to pause streaming on adding known faces
current_frame = None

# === Face Detection and Emotion Model (DeepFace module) ===
def detect_emotion(face_img):
    try:
        analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion'].capitalize()
    except Exception:
        return "Neutral"

# === Anti-spoofing based on texture: higher LBP uniformity and contrast → real ===
def is_live_face(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_var = np.var(lbp)
        contrast = gray.std()
        # Heuristic thresholds: adjust based on experimentation
        if lbp_var > 1.5 and contrast > 25:
            return True
        else:
            return False
    except Exception as e:
        print("Spoof detection error:", e)
        return False

# === Save known face snapshot with annotation ===
def save_snapshot(name, emotion, face_img):
    # Snapshot image file are named with the name and timestamp for uniqueness
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    snapshot_file = os.path.join(snapshot_dir, f"{name}_{timestamp}.jpg")
    cv2.imwrite(snapshot_file, face_img)
    print(f"📸 Saved snapshot: {snapshot_file}")

# === Handle unknown face prompt ===
def prompt_unknown_face(encoding, face_img):
    global paused
    paused = True
    # Submit new face to the list of known id
    def on_submit():
        person_name = name_input.get()
        if person_name:
            face_file = os.path.join(known_faces_dir, f"{person_name}.jpg")
            cv2.imwrite(face_file, face_img)
            known_face_encodings.append(encoding)
            known_face_names.append(person_name)
            saved_known_faces.add(person_name)
            print(f"✅ Saved new identity: {person_name}")
        prompt_win.destroy()
        resume_stream()
    # Cancel the prompting
    def on_cancel():
        prompt_win.destroy()
        resume_stream()
    # UI setup, components layout
    prompt_win = tk.Toplevel(root)
    prompt_win.title("Unknown Face Detected")
    tk.Label(prompt_win, text="Enter this person's name:").pack(pady=5)
    name_input = tk.Entry(prompt_win)
    name_input.pack(pady=5)
    tk.Button(prompt_win, text="OK", command=on_submit).pack(side='left', padx=10, pady=10)
    tk.Button(prompt_win, text="Cancel", command=on_cancel).pack(side='right', padx=10, pady=10)

# Resume post-prompting new id
def resume_stream():
    global paused
    paused = False

# === Main Update Loop ===
def update_frame():
    global current_frame, paused
    # Process frames (to RGB)
    if not paused:
        ret, frame = video.read()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frame = frame.copy()
        # Determine location similarity
        face_locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, face_locations)
        # Compare similarity
        for (top, right, bottom, left), encoding in zip(face_locations, encodings):
            name = "Unknown"
            distances = face_recognition.face_distance(known_face_encodings, encoding)
            # Apply static heuristic distance threshold (50%)
            if len(distances) > 0 and np.min(distances) < 0.8:
                match = np.argmin(distances)
                name = known_face_names[match]
            # Crop faces to be saved
            face_crop = frame[top:bottom, left:right]
            emotion = detect_emotion(face_crop)
            # Perform liveness check
            live = is_live_face(face_crop)
            # Annotation: name + emotion
            label = f"{name} - {emotion}"
            box_color = (0, 255, 0) if live else (0, 0, 255)  # Green = live, Red = spoof
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), box_color, cv2.FILLED)
            cv2.putText(frame, label, (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # Save snapshot of the known one only ONCE
            if name != "Unknown" and name not in saved_known_faces:
                save_snapshot(name, emotion, face_crop)
                saved_known_faces.add(name)
            elif name == "Unknown":
                uid = str(hash(tuple(encoding.round(4))))
                if uid not in unknown_faces:
                    unknown_faces[uid] = {
                        "encoding": encoding,
                        "image": face_crop
                    }
                    prompt_unknown_face(encoding, face_crop)
        # UI configuration
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk 
        video_label.configure(image=imgtk)
    # Streaming pace for detection (lower FPS for lighter load but lower the smoothness)
    root.after(16, update_frame)  # Aim for ~60 FPS

# === Quit Button Handler (UI) ===
def stop_app():
    video.release()
    root.destroy()

# === GUI Setup (Tkinter) ===
root = tk.Tk()
root.title("Face Recognition + Emotion")
video_label = tk.Label(root)
video_label.pack()
stop_btn = tk.Button(root, text="Stop", command=stop_app, bg='red', fg='white')
stop_btn.pack(pady=10)

# === Start webcam ===
video = cv2.VideoCapture(0)
update_frame()
root.mainloop()
