"""
----------------------------------------------------------------------------
 Face-Recognition Attendance System  (MediaPipe + MobileFaceNet + SilentFace)
----------------------------------------------------------------------------
 * Face detection ............... MediaPipe Face Detection (CPU-friendly)
 * Face embedding ............... MobileFaceNet (ONNX) via onnxruntime
 * Face matching ................ Cosine similarity against saved embeddings
 * Emotion analysis ............. DeepFace (facial_expression_model_weights)
 * Anti-spoofing ................ Silent-Face ensemble (3 tiny CNNs)
 * GUI .......................... Tkinter / ttk with streamlined styling
----------------------------------------------------------------------------
 Author :  YOU
 Date   :  2025-05-20
----------------------------------------------------------------------------"""

# --------------------------------------------------------------------------
# 1. Imports
# --------------------------------------------------------------------------
import cv2                              # OpenCV for I/O and image ops
import mediapipe as mp                  # MediaPipe for lightweight face-detection
import onnxruntime as ort               # Run MobileFaceNet 
from deepface import DeepFace           # Emotion recognition
import numpy as np
import os, time, math, platform
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

# Silent-Face files (copy from repo into the project folder)
from spoof_detector import SpoofDetector  # thin wrapper around anti_spoof_predict.py

# --------------------------------------------------------------------------
# 2. Paths & global folders
# --------------------------------------------------------------------------
KNOWN_DIR   = Path("known_faces")
SNAP_DIR    = Path("snapshot")
from pathlib import Path

MODEL_DIR  = Path("resources/anti_spoof_models") # ← copied from Silent-Face repo
ONNX_MODEL = Path("mobilefacenet.onnx")          # ← download or place next to script

# Ensure directory exist and valid
for p in (KNOWN_DIR, SNAP_DIR):
    p.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# 3. Helper: load MobileFaceNet once
# --------------------------------------------------------------------------
''' MobileFaceNet expects 112×112 RGB images, float32, range ≈ -1 … 1 '''
ort_sess = ort.InferenceSession(str(ONNX_MODEL), providers=['CPUExecutionProvider'])
input_name = ort_sess.get_inputs()[0].name

def embed_face(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR crop → 128-D L2-normalised embedding.
    """
    face_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_res = cv2.resize(face_rgb, (112, 112)).astype(np.float32)
    face_res = (face_res - 127.5) / 128.0
    blob = np.transpose(face_res, (2, 0, 1))[None]  # (1,3,112,112)
    emb = ort_sess.run(None, {input_name: blob})[0][0]
    return emb / np.linalg.norm(emb)

# --------------------------------------------------------------------------
# 4. Load all saved embeddings (if any) from images in known_faces/
# --------------------------------------------------------------------------
known_embeddings, known_names = [], []
# Read image and embed
for img_path in KNOWN_DIR.glob("*.[pjPJ][pnPN][gG]"):   # jpg / png case-insensitive
    img = cv2.imread(str(img_path))
    if img is None: continue
    emb = embed_face(img)
    known_embeddings.append(emb)
    known_names.append(img_path.stem)  # filename without extension
known_embeddings = np.array(known_embeddings)  # shape (N,128)

# --------------------------------------------------------------------------
# 5. Initialise MediaPipe face detector (single-image mode)
# --------------------------------------------------------------------------
mp_detect = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# --------------------------------------------------------------------------
# 6. Silent-Face anti-spoofing ensemble
# --------------------------------------------------------------------------
spoof_detector = SpoofDetector(model_dir=str(MODEL_DIR))

# --------------------------------------------------------------------------
# 7. Tkinter-based GUI boilerplate
# --------------------------------------------------------------------------
root = tk.Tk()
root.title("Real-Time Face Attendance + Emotion + Anti-Spoofing")
root.geometry("960x720")
root.configure(bg="#2b2b2b")

# -- Styles -----------------------------------------------------------------
style = ttk.Style(root)
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 12), padding=6)
style.configure("Info.TLabel", background="#2b2b2b", foreground="#e1e1e1",
                font=("Segoe UI", 11))

# -- Main video canvas ------------------------------------------------------
video_lbl = ttk.Label(root)
video_lbl.pack(padx=10, pady=10)

# -- Status bar -------------------------------------------------------------
status_var = tk.StringVar(value="🔄 Initialising camera…")
status_bar = ttk.Label(root, textvariable=status_var, style="Info.TLabel")
status_bar.pack(fill="x", padx=5, pady=(0, 10))

# -- Control buttons --------------------------------------------------------
btn_frame = ttk.Frame(root)
btn_frame.pack(pady=5)
# Terminate session on quit
def quit_app():
    cap.release()
    root.destroy()
# Button styles
quit_btn = ttk.Button(btn_frame, text="Quit", command=quit_app)
quit_btn.grid(row=0, column=0, padx=5)

# --------------------------------------------------------------------------
# 8. Utilities
# --------------------------------------------------------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-5))
# Check similarity threshold (55%)
def match_embedding(emb: np.ndarray, thresh: float = 0.55):
    """
    Return (name, score) or ("Unknown", None)
    """
    if known_embeddings.size == 0:
        return "Unknown", None
    sims = np.dot(known_embeddings, emb)  # embeddings are already L2-norm
    idx = int(np.argmax(sims))
    if sims[idx] > thresh:
        return known_names[idx], sims[idx]
    return "Unknown", None
# Save snapshot of known face to folder
def save_snapshot(name, emotion, img_bgr):
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = SNAP_DIR / f"{name}_{emotion}_{ts}.jpg"
    cv2.imwrite(str(path), img_bgr)
    print(f"📸 Saved {path}")

# --------------------------------------------------------------------------
# 9. Main frame-update loop
# --------------------------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if platform.system()=="Windows" else 0)
paused   = False
saved_one_per_name = set()
# Continuously refresh frame
def update():
    global paused
    if paused:
        root.after(50, update)
        return
    # Read frame
    ret, frame = cap.read()
    if not ret:
        status_var.set("⚠️  Camera read error")
        root.after(200, update)
        return
    # Convert color ensure RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = mp_detect.process(frame_rgb).detections
    if detections is None:
        # No face in this frame
        draw = frame.copy()
    else:
        draw = frame.copy()
        for det in detections:
            # Convert bbox from relative → absolute px
            h, w, _ = frame.shape
            box = det.location_data.relative_bounding_box
            x1, y1 = int(box.xmin * w), int(box.ymin * h)
            bw, bh  = int(box.width * w), int(box.height * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x1 + bw), min(h, y1 + bh)
            # Define bbox around the face
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            # 1️. Embedding & match
            emb = embed_face(face_crop)
            name, score = match_embedding(emb)
            # 2️. Emotion
            try:
                emot = DeepFace.analyze(face_crop, actions=['emotion'],
                                        enforce_detection=False)[0]['dominant_emotion'].capitalize()
            except Exception:
                emot = "Neutral"
            # 3️. Anti-spoof
            is_real, conf = spoof_detector.check_spoof(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            # 4️. Visuals
            color = (0,255,0) if is_real else (0,0,255)
            cv2.rectangle(draw, (x1,y1), (x2,y2), color, 2)
            label = f"{name} | {emot} | {'Live' if is_real else 'Spoof'}"
            cv2.putText(draw, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
            # 5. Save new snapshots / unknown prompts
            if name != "Unknown" and name not in saved_one_per_name:
                save_snapshot(name, emot, face_crop)
                saved_one_per_name.add(name)
            elif name == "Unknown" and conf and is_real:
                prompt_unknown(face_crop, emb)
    # Show frame ------------------------------------------------------------
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)))
    video_lbl.configure(image=img_tk)
    video_lbl.image = img_tk  # keep ref
    # Set status
    status_var.set(f"Faces: {len(detections or [])}   Known IDs: {len(known_names)}")
    root.after(16, update)  # ~60 fps

# --------------------------------------------------------------------------
# 10. Prompt for unknown face – adds new ID to database
# --------------------------------------------------------------------------
def prompt_unknown(face_img_bgr, emb):
    global paused
    paused = True
    # Convert preview to PhotoImage
    preview = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
    pil_prev = Image.fromarray(preview)
    pil_prev = pil_prev.resize((120, 120))
    img_prev = ImageTk.PhotoImage(image=pil_prev)
    # Prompt title
    win = tk.Toplevel(root)
    win.title("Add New Person")
    win.configure(bg="#383838")
    # Label text
    ttk.Label(win, text="New face detected!\nEnter name:",
              style="Info.TLabel").pack(pady=5)
    # Append label
    lbl_img = ttk.Label(win, image=img_prev)
    lbl_img.image = img_prev
    lbl_img.pack(pady=5)
    # Modal window stylings
    entry = ttk.Entry(win, width=25)
    entry.pack(pady=5)
    entry.focus()
    # Save new id and close prompt window
    def save_and_close():
        name = entry.get().strip()
        if not name:
            messagebox.showwarning("Name missing", "Please enter a non-empty name.")
            return
        # Save original crop
        out_path = KNOWN_DIR / f"{name}.jpg"
        cv2.imwrite(str(out_path), face_img_bgr)
        # Update memory
        known_names.append(name)
        known_embeddings.resize((known_embeddings.shape[0]+1, 128), refcheck=False)
        known_embeddings[-1] = emb
        messagebox.showinfo("Success", f"Added {name} to database.")
        win.destroy()
        resume()
    # Button styles
    ttk.Button(win, text="Add", command=save_and_close).pack(side="left", padx=15, pady=10)
    ttk.Button(win, text="Cancel", command=resume).pack(side="right", padx=15, pady=10)

# Cancel prmpt
def resume():
    global paused
    paused = False

# --------------------------------------------------------------------------
# 11. Kick-off
# --------------------------------------------------------------------------
update()
root.mainloop()
