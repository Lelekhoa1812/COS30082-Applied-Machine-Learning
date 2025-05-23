# inference.py

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# -----------------------------------------------------------------------------
# 1. Redefine exactly the architectures you used at training time
# -----------------------------------------------------------------------------
class FaceBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1),     nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1),    nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x):
        return self.conv(x).view(x.size(0), -1)

class FaceClassifier(nn.Module):
    def __init__(self, num_classes, embed_dim=128):
        super().__init__()
        self.backbone      = FaceBackbone()
        self.embed_head    = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        self.classify_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        e = self.embed_head(f)
        return self.classify_head(e), F.normalize(e, dim=1)

# -----------------------------------------------------------------------------
# 2. Setup device and transforms (must match training)
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

# -----------------------------------------------------------------------------
# 3. Load the saved classification model weights
# -----------------------------------------------------------------------------
NUM_CLASSES = 500  # change to whatever you trained with
clf_model = FaceClassifier(NUM_CLASSES).to(DEVICE)
clf_model.load_state_dict(torch.load("clf_model1.pth", map_location=DEVICE))
clf_model.eval()
print("Loaded clf_model1.pth successfully")

# -----------------------------------------------------------------------------
# 4. Helper to extract a normalized embedding from an image
# -----------------------------------------------------------------------------
def get_classification_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    x   = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, emb = clf_model(x)
    return emb.cpu().numpy()[0]

# -----------------------------------------------------------------------------
# 5. Example usage: compare two images
# -----------------------------------------------------------------------------
img1 = "baby1.jpg"
img2 = "justin.jpg"

emb1 = get_classification_embedding(img1)
emb2 = get_classification_embedding(img2)
sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"Cosine similarity (classification model): {sim:.4f}")

THRESHOLD = 0.6
if sim > THRESHOLD:
    print("✅ Same person")
else:
    print("❌ Different people")

# -----------------------------------------------------------------------------
# 6. (Optional) Live webcam face verification using classification embeddings
# -----------------------------------------------------------------------------
def webcam_verify(ref_image_path):
    # compute reference embedding once
    ref_emb = get_classification_embedding(ref_image_path)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        for x,y,w,h in faces:
            face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(face)
            emb  = get_classification_embedding(pil)  # you can modify helper to accept PIL
            sim  = np.dot(emb, ref_emb)/(np.linalg.norm(emb)*np.linalg.norm(ref_emb))
            label, color = ("Target", (0,255,0)) if sim>THRESHOLD else ("Unknown",(0,0,255))
            cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)
            cv2.putText(frame, f"{label} {sim:.2f}", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Webcam Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# To run webcam verification on, e.g., "baby1.jpg":
# webcam_verify("baby1.jpg")
