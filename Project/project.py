"""
----------------------------------------------------------------------------
 Face-Recognition Attendance System  (Classifier / Triplet Loss + DeepFace + SilentFace)
----------------------------------------------------------------------------
 * Face detection ............... Classifier + Triplet Loss Model
 * Face embedding ............... Embedding head-layer of the CNN model
 * Face bbox frame .............. FaceDetector (Haar Cascade) detects faces from frames
 * Face matching ................ Cosine/Euclidean similarity against saved embeddings
 * Emotion analysis ............. DeepFace (facial_expression_model_weights)
 * Anti-spoofing ................ Silent-Face ensemble (2 tiny CNNs)
 * GUI .......................... Tkinter / ttk with streamlined styling
----------------------------------------------------------------------------
 Author :  Dang Khoa Le
 Date   :  2025-05-25
----------------------------------------------------------------------------"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision import transforms
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os, time, platform
from pathlib import Path
from deepface import DeepFace
from spoof_detector import SpoofDetector

# ----------------------------------------------------------------------------
# 1. Model definitions (Triplet & Classification)
# ----------------------------------------------------------------------------
class BaseCNN(nn.Module):
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

class Classifier(nn.Module):
    def __init__(self, num_classes, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone      = BaseCNN()
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
tri_model = nn.Sequential(BaseCNN(), embed_head).to(DEVICE)
tri_model.load_state_dict(torch.load('models/tri_model.pth', map_location=DEVICE))
tri_model.eval()
print('✅ Loaded Triplet Loss Model')

# Classification model (set NUM_CLASSES to your training setup)
NUM_CLASSES = 500
cls_model = Classifier(NUM_CLASSES).to(DEVICE)
cls_model.load_state_dict(torch.load('models/cls_model.pth', map_location=DEVICE))
cls_model.eval()
print('✅ Loaded Classifier Model')

# ----------------------------------------------------------------------------
# 3. Transforms for both models (matching training)
# ----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
# Folder to store embedding vectors
EMBED_DIR = Path("embeddings")
EMBED_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# 4. Embedding helpers
# ----------------------------------------------------------------------------
def embed_triplet(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): emb = tri_model(x)
    emb = emb.cpu().numpy()[0]
    return emb / np.linalg.norm(emb)

def embed_classification(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): _, emb = cls_model(x)
    emb = emb.cpu().numpy()[0]
    return emb / np.linalg.norm(emb)


# ----------------------------------------------------------------------------
# 5. Load known-face embeddings (with save + print)
# ----------------------------------------------------------------------------
KNOWN_DIR = Path('known_faces')
KNOWN_DIR.mkdir(exist_ok=True)
# Stack ids and vectors
known_names = []; known_emb_trip = []; known_emb_cls  = []
# Validate image format
for img_path in KNOWN_DIR.glob('*.[pjPJ][pnPN][gG]'):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[⚠️] Skipping unreadable image: {img_path}")
        continue
    # Extract label
    name = img_path.stem
    print(f"[INFO] Embedding '{name}'...")
    # Embedding of the 2 models separately
    emb_trip = embed_triplet(img)
    emb_cls  = embed_classification(img)
    # Append to known face list
    known_names.append(name)
    known_emb_trip.append(emb_trip)
    known_emb_cls.append(emb_cls)
    # Save embeddings to disk for future debugging or reuse
    np.save(EMBED_DIR / f"{name}_triplet.npy", emb_trip)
    np.save(EMBED_DIR / f"{name}_cls.npy", emb_cls)
print(f"[✅] Loaded and saved embeddings for {len(known_names)} known faces.")
# Stack into numpy arrays
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

# Right-side model switch buttons
switch_frame = ttk.Frame(root)
switch_frame.pack(side='right', fill='y', padx=10, pady=20)

# Video canvas
display = ttk.Label(root)
display.pack(side='left', fill='both', expand=True, padx=10, pady=10)

# Status bar
status_var = tk.StringVar(value='Model: Triplet')
status_bar = ttk.Label(root, textvariable=status_var, style='Info.TLabel')
status_bar.pack(fill='x', padx=5, pady=(0,10))

current_model = 'triplet' # Default model (also better one)
# Change model on selection
def set_model(mode):
    global current_model
    current_model = mode
    status_var.set(f'Model: {mode.capitalize()}')

# Toggle model changing
btn_trip = ttk.Button(switch_frame, text='Triplet Model', command=lambda: set_model('triplet')); btn_trip.pack(pady=5)
btn_cls  = ttk.Button(switch_frame, text='Classification Model', command=lambda: set_model('classification')); btn_cls.pack(pady=5)

# Open gallery window, showing list of known ids and snapshots
def open_gallery(dir_path, title):
    win = tk.Toplevel(root)
    win.title(title)
    canvas = tk.Canvas(win)
    scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    # Create canvas
    canvas.create_window((10, 10), window=scroll_frame, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)
    # Stack images loaded from local directory
    images = []
    for i, img_file in enumerate(Path(dir_path).glob('*')):
        try:
            img = Image.open(img_file).resize((70,100))
            tk_img = ImageTk.PhotoImage(img)
        except:
            continue
        # Initialise 3x3 grid with images and scrolling 
        lbl_img = ttk.Label(scroll_frame, image=tk_img)
        lbl_img.image = tk_img
        lbl_img.grid(row=i//3*2, column=i%3, padx=5, pady=5)
        lbl_text = ttk.Label(scroll_frame, text=img_file.stem, style='Info.TLabel')
        lbl_text.grid(row=i//3*2+1, column=i%3, padx=5, pady=(0,10))
    # Close window button
    close_btn = ttk.Button(win, text='Close', command=win.destroy)
    close_btn.pack(side='bottom', pady=10)
    canvas.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

# Configure buttons
btn_known = ttk.Button(switch_frame, text='Known IDs', command=lambda: open_gallery(KNOWN_DIR, 'Known Faces'))
btn_known.pack(pady=(20,5))
btn_snap  = ttk.Button(switch_frame, text='Snapshot', command=lambda: open_gallery('snapshot', 'Snapshot Gallery'))
btn_snap.pack(pady=5)

# Add new id manually, trigger the prompt_unknown function and window from file upload
def handle_add_id():
    file_path = filedialog.askopenfilename(title='Select face image', filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return
    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Failed to load the selected image.")
        return
    prompt_unknown(img, embed_triplet(img) if current_model == 'triplet' else embed_classification(img))

# Add ID Upload Button
btn_add_id = ttk.Button(switch_frame, text='Add ID (Upload)', command=handle_add_id)
btn_add_id.pack(pady=(10, 5))

# Quit button
btn_quit = ttk.Button(switch_frame, text='Quit', command=lambda: (cap.release(), root.destroy()))
btn_quit.pack(pady=20)


# --------------------------------------------------------------------------
# Matching utils
# --------------------------------------------------------------------------
def match_embedding(emb: np.ndarray, thresh: float = 0.5):
    known_emb = known_emb_trip if current_model=='triplet' else known_emb_cls
    if known_emb.size == 0:
        return 'Unknown', None
    sims = np.dot(known_emb, emb)                   # cosine similarity
    idx = int(np.argmax(sims))
    # sims = np.linalg.norm(known_emb - emb, axis=1) # euclidean distance
    # idx = int(np.argmin(sims))
    if sims[idx] > thresh:
        return known_names[idx], float(sims[idx])
    return 'Unknown', None

# Snapshot dir
SNAP_DIR = Path('snapshot'); SNAP_DIR.mkdir(exist_ok=True)
# Save snapshot of the known id only once, format filename
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

# Continuously refresh frame with updates and labelling
def update():
    global paused, last_unknown_time
    if paused:
        root.after(50, update); return
    ret, frame = cap.read()
    if not ret:
        status_var.set('⚠️ Camera error'); root.after(200, update); return
    # Image processing - Locate bbox
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    disp = frame.copy()

    # Prepare label from bbox extracted
    for (x,y,w,h) in faces:
        crop = frame[y:y+h, x:x+w]
        emb = (embed_triplet(crop) if current_model=='triplet' else embed_classification(crop))
        name, score = match_embedding(emb)
        # Feature 1: Emotion
        try:
            emot = DeepFace.analyze(crop, actions=['emotion'], enforce_detection=False)[0]['dominant_emotion'].capitalize()
        except:
            emot = 'Neutral'
        # Feature 2: Anti-spoof
        is_real, conf = spoof_detector.check_spoof(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) 
        color = (0,255,0) if is_real else (0,0,255) # Can customize confidence level check 
        cv2.rectangle(disp, (x,y), (x+w, y+h), color, 2)
        # --- Fallback if score is invalid ---
        if not isinstance(score, (int, float)):
            score = 0.0
            st_score = "N/A"
        else:
            st_score = f"{score:.4f}"        
        # --- Color change on confident threshold ---
        if score < 0.3:   score_color = (0, 0, 255)    # Red
        elif score < 0.5: score_color = (0, 165, 255)  # Orange
        elif score < 0.7: score_color = (255, 0, 0)    # Blue
        else:             score_color = (128, 0, 128)  # Purple
        # --- Text parts ---
        label_name = f"{name},"
        label_emot = f" | {emot} | {'Live' if is_real else 'Spoof'}"
        # --- Text positioning ---
        text_start = (x, y - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        # --- Compute text sizes to place colored parts ---
        size_name, _ = cv2.getTextSize(label_name, font, font_scale, thickness)
        size_score, _ = cv2.getTextSize(st_score, font, font_scale, thickness)
        # Draw each component separately with its color
        cv2.putText(disp, label_name, text_start, font, font_scale, color, thickness)
        cv2.putText(disp, st_score, (text_start[0] + size_name[0], text_start[1]), font, font_scale, score_color, thickness)
        cv2.putText(disp, label_emot, 
                    (text_start[0] + size_name[0] + size_score[0], text_start[1]),
                    font, font_scale, color, thickness)
        # Snapshot Saver
        if name!='Unknown' and name not in saved_names:
            save_snapshot(name, emot, crop); saved_names.add(name)
        elif name == 'Unknown' and is_real:
            now = time.time()
            if now - last_unknown_time > UNKNOWN_PROMPT_INTERVAL:
                last_unknown_time = now    # ← update the global
                prompt_unknown(crop, emb)

    # Video display overlay
    img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
    display.configure(image=img_tk); display.image = img_tk
    status_bar.config(text=f"Model: {current_model.capitalize()} | Faces: {len(faces)} | Known: {len(known_names)}")
    root.after(16, update) # 60 FPS


# --------------------------------------------------------------------------
# Unknown id prompt
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
# Continue after action
def resume():
    global paused; paused = False

# Start
update()
root.mainloop()