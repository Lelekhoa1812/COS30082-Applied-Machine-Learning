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
 Author :  Dang Khoa Le
 Date   :  2025-05-20
----------------------------------------------------------------------------"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision import transforms
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import os, time, platform
from pathlib import Path
from deepface import DeepFace
from spoof_detector import SpoofDetector

# ----------------------------------------------------------------------------
# 1. Model definitions (Triplet & Classification)
# ----------------------------------------------------------------------------
class FaceBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x):
        return self.conv(x).view(x.size(0), -1)

# Triplet embedding head
EMBED_DIM = 128
embed_head = nn.Sequential(
    nn.Linear(128, EMBED_DIM),
    nn.BatchNorm1d(EMBED_DIM),
    nn.ReLU(),
)

class FaceClassifier(nn.Module):
    def __init__(self, num_classes, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone      = FaceBackbone()
        self.embed_head   = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        self.classify_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        e = self.embed_head(f)
        return self.classify_head(e), F.normalize(e, dim=1)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------------
# 2. Load models
# ----------------------------------------------------------------------------
# Triplet model
triplet_model = nn.Sequential(FaceBackbone(), embed_head).to(DEVICE)
triplet_model.load_state_dict(torch.load('triplet_model1.pth', map_location=DEVICE))
triplet_model.eval()
print('✅ Loaded triplet_model1.pth')

# Classification model (set NUM_CLASSES to your training setup)
NUM_CLASSES = 500
clf_model = FaceClassifier(NUM_CLASSES).to(DEVICE)
clf_model.load_state_dict(torch.load('clf_model1.pth', map_location=DEVICE))
clf_model.eval()
print('✅ Loaded clf_model1.pth')

# ----------------------------------------------------------------------------
# 3. Transforms for both models (matching training)
# ----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# ----------------------------------------------------------------------------
# 4. Embedding helpers
# ----------------------------------------------------------------------------
def embed_triplet(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): emb = triplet_model(x)
    emb = emb.cpu().numpy()[0]
    return emb / np.linalg.norm(emb)


def embed_classification(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): _, emb = clf_model(x)
    emb = emb.cpu().numpy()[0]
    return emb / np.linalg.norm(emb)

# ----------------------------------------------------------------------------
# 5. Load known-face embeddings
# ----------------------------------------------------------------------------
KNOWN_DIR = Path('known_faces')
KNOWN_DIR.mkdir(exist_ok=True)
known_names = []
known_emb_trip = []
known_emb_cls  = []
for img_path in KNOWN_DIR.glob('*.[pjPJ][pnPN][gG]'):
    img = cv2.imread(str(img_path))
    if img is None: continue
    name = img_path.stem
    known_names.append(name)
    known_emb_trip.append(embed_triplet(img))
    known_emb_cls.append(embed_classification(img))
known_emb_trip = np.array(known_emb_trip)
known_emb_cls  = np.array(known_emb_cls)

# ----------------------------------------------------------------------------
# 6. Face detector (OpenCV Haar Cascade)
# ----------------------------------------------------------------------------
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ----------------------------------------------------------------------------
# 7. Anti-spoof and emotion
# ----------------------------------------------------------------------------
spoof_detector = SpoofDetector(model_dir='resources/anti_spoof_models')

# ----------------------------------------------------------------------------
# 8. Tkinter GUI
# ----------------------------------------------------------------------------
root = tk.Tk()
root.title('Face Attendance System')
root.geometry('1024x768')
root.configure(bg='#2b2b2b')

style = ttk.Style(root)
style.theme_use('clam')
style.configure('TButton', font=('Segoe UI',12), padding=6)
style.configure('Info.TLabel', background='#2b2b2b', foreground='#e1e1e1', font=('Segoe UI',11))

# Video canvas
display = ttk.Label(root)
display.pack(padx=10, pady=10)

# Status bar
status_var = tk.StringVar(value='Model: Triplet')
status_bar = ttk.Label(root, textvariable=status_var, style='Info.TLabel')
status_bar.pack(fill='x', padx=5, pady=(0,10))

# Right-side model switch buttons
switch_frame = ttk.Frame(root)
switch_frame.pack(side='right', fill='y', padx=10, pady=20)

current_model = 'triplet'

def set_model(mode):
    global current_model
    current_model = mode
    status_var.set(f'Model: {mode.capitalize()}')

btn_trip = ttk.Button(switch_frame, text='Triplet Model', command=lambda: set_model('triplet'))
btn_trip.pack(pady=5)
btn_cls  = ttk.Button(switch_frame, text='Classification Model', command=lambda: set_model('classification'))
btn_cls.pack(pady=5)

# Quit button
btn_quit = ttk.Button(switch_frame, text='Quit', command=lambda: (cap.release(), root.destroy()))
btn_quit.pack(pady=20)

# --------------------------------------------------------------------------
# Matching util
# --------------------------------------------------------------------------
def match_embedding(emb: np.ndarray, thresh: float = 0.8):
    known_emb = known_emb_trip if current_model=='triplet' else known_emb_cls
    if known_emb.size == 0:
        return 'Unknown', None
    sims = np.dot(known_emb, emb)
    idx = int(np.argmax(sims))
    if sims[idx] > thresh:
        return known_names[idx], float(sims[idx])
    return 'Unknown', None

# Snapshot dir
SNAP_DIR = Path('snapshot'); SNAP_DIR.mkdir(exist_ok=True)

def save_snapshot(name, emotion, img_bgr):
    ts = time.strftime('%Y%m%d-%H%M%S')
    path = SNAP_DIR / f"{name}_{emotion}_{ts}.jpg"
    cv2.imwrite(str(path), img_bgr)

# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if platform.system()=='Windows' else 0)
paused = False
saved_names = set()
last_unknown_time   = 0.0
UNKNOWN_PROMPT_INTERVAL = 10.0  # seconds

def update():
    global paused, last_unknown_time
    if paused:
        root.after(50, update); return
    ret, frame = cap.read()
    if not ret:
        status_var.set('⚠️ Camera error'); root.after(200, update); return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    disp = frame.copy()

    for (x,y,w,h) in faces:
        crop = frame[y:y+h, x:x+w]
        emb = (embed_triplet(crop) if current_model=='triplet' else embed_classification(crop))
        name, score = match_embedding(emb)
        # emotion
        try:
            emot = DeepFace.analyze(crop, actions=['emotion'], enforce_detection=False)[0]['dominant_emotion'].capitalize()
        except:
            emot = 'Neutral'
        # anti-spoof
        is_real, conf = spoof_detector.check_spoof(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        color = (0,255,0) if is_real else (0,0,255)
        cv2.rectangle(disp, (x,y), (x+w, y+h), color, 2)
        cv2.putText(disp, f"{name} | {emot} | {'Live' if is_real else 'Spoof'}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # snapshot
        if name!='Unknown' and name not in saved_names:
            save_snapshot(name, emot, crop); saved_names.add(name)
        elif name == 'Unknown' and is_real:
            now = time.time()
            if now - last_unknown_time > UNKNOWN_PROMPT_INTERVAL:
                last_unknown_time = now    # ← update the global
                prompt_unknown(crop, emb)


    img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
    display.configure(image=img_tk); display.image = img_tk
    status_bar.config(text=f"Model: {current_model.capitalize()} | Faces: {len(faces)} | Known: {len(known_names)}")
    root.after(12, update)

# --------------------------------------------------------------------------
# Unknown prompt
# --------------------------------------------------------------------------
def prompt_unknown(crop, emb):
    global paused
    paused = True
    preview = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((120,120)))
    win = tk.Toplevel(root); win.title('Add New Person'); win.configure(bg='#383838')
    ttk.Label(win, text='New face detected! Enter name:', style='Info.TLabel').pack(pady=5)
    lbl = ttk.Label(win, image=preview); lbl.image = preview; lbl.pack(pady=5)
    entry = ttk.Entry(win, width=25); entry.pack(pady=5); entry.focus()
    def save_and_close():
        name = entry.get().strip() or None
        if not name: messagebox.showwarning('Name missing','Please enter a name.'); return
        path = KNOWN_DIR / f"{name}.jpg"
        cv2.imwrite(str(path), crop)
        known_names.append(name)
        # update embeddings
        known_emb = embed_triplet(crop) if current_model=='triplet' else embed_classification(crop)
        if current_model=='triplet':
            global known_emb_trip; known_emb_trip = np.vstack([known_emb_trip, known_emb])
        else:
            global known_emb_cls;  known_emb_cls  = np.vstack([known_emb_cls,  known_emb])
        messagebox.showinfo('Added', f'Added {name}')
        win.destroy(); resume()
    btnf = ttk.Frame(win); btnf.pack(pady=10)
    ttk.Button(btnf, text='Add', command=save_and_close).pack(side='left', padx=5)
    ttk.Button(btnf, text='Cancel', command=lambda: (win.destroy(), resume())).pack(side='right', padx=5)

def resume():
    global paused; paused = False

# Start
update()
root.mainloop()