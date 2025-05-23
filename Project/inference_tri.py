# inference.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

# -----------------------------------------------------------------------------
# 1) Redefine exactly the same nn.Sequential(backbone, embed_head) you used at save time
# -----------------------------------------------------------------------------
class FaceBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
    def forward(self,x):
        return self.conv(x).view(x.size(0), -1)

# Here: embed_head must be a plain Sequential, *not* wrapped in a submodule named `net`
EMBED_DIM = 128
embed_head = nn.Sequential(
    nn.Linear(128, EMBED_DIM),
    nn.BatchNorm1d(EMBED_DIM),
    nn.ReLU(),
)

# Build the exact same structure as you originally saved
triplet_model = nn.Sequential(
    FaceBackbone(),
    embed_head
)

# -----------------------------------------------------------------------------
# 2) Load the weights
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
triplet_model.to(device)
triplet_model.load_state_dict(torch.load("triplet_model1.pth", map_location=device))
triplet_model.eval()
print("Loaded triplet_model1.pth successfully")

# -----------------------------------------------------------------------------
# 3) Prep your transforms
# -----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

# -----------------------------------------------------------------------------
# 4) Helper to extract an embedding
# -----------------------------------------------------------------------------
def get_triplet_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    x   = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = triplet_model(x)
    return emb.cpu().numpy()[0]

# -----------------------------------------------------------------------------
# 5) Quick smoke test
# -----------------------------------------------------------------------------
a = get_triplet_embedding("baby1.jpg")
b = get_triplet_embedding("baby.jpg")
sim = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
print("Cosine similarity:", sim)
THRESHOLD = 0.6
if sim > THRESHOLD:
    print("✅ Same person")
else:
    print("❌ Different people")
def is_same_person(img1, img2, threshold=0.6):
    e1 = get_triplet_embedding(img1)
    e2 = get_triplet_embedding(img2)
    sim = np.dot(e1,e2)/(np.linalg.norm(e1)*np.linalg.norm(e2))
    return sim, sim > threshold
