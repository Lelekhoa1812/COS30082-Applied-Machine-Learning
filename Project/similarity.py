import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Model Definition (Triplet)
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

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load triplet model
tri_model = nn.Sequential(BaseCNN(), embed_head).to(DEVICE)
tri_model.load_state_dict(torch.load('models/tri_model.pth', map_location=DEVICE))
tri_model.eval()
print('✅ Loaded Triplet Loss Model')

# Image transformation
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# ----------------------------------------------------------------------------
# Embedding and Comparison Functions
# ----------------------------------------------------------------------------
def get_embedding(img_bgr: np.ndarray) -> np.ndarray:
    """Convert image to embedding vector using triplet model"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): 
        emb = tri_model(x)
    emb = emb.cpu().numpy()[0]
    return emb / np.linalg.norm(emb)  # Normalize

def compare_images(test_embedding, sample_embedding):
    """Calculate cosine similarity between two embeddings"""
    return np.dot(test_embedding, sample_embedding)

# ----------------------------------------------------------------------------
# GUI for Image Upload and Comparison
# ----------------------------------------------------------------------------
class ImageComparator:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Similarity Comparator")
        self.root.geometry("600x400")
        
        # Sample images folder
        self.sample_folder = "sample"
        if not os.path.exists(self.sample_folder):
            os.makedirs(self.sample_folder)
            messagebox.showinfo("Info", f"Created '{self.sample_folder}' folder. Please add sample images there.")
        
        # GUI Elements
        self.label = tk.Label(root, text="Upload an image to compare with samples:", font=('Arial', 12))
        self.label.pack(pady=20)
        
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image, font=('Arial', 12))
        self.upload_btn.pack(pady=10)
        
        self.result_text = tk.Text(root, height=15, width=70)
        self.result_text.pack(pady=20, padx=20)
        self.result_text.insert(tk.END, "Results will appear here...\n")
        self.result_text.config(state=tk.DISABLED)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image to Compare",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
        
        try:
            test_img = cv2.imread(file_path)
            if test_img is None:
                raise ValueError("Could not read image file")
            
            test_embedding = get_embedding(test_img)
            results = []
            
            # Compare with each sample image
            for sample_file in os.listdir(self.sample_folder):
                if sample_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_path = os.path.join(self.sample_folder, sample_file)
                    sample_img = cv2.imread(sample_path)
                    
                    if sample_img is not None:
                        sample_embedding = get_embedding(sample_img)
                        similarity = compare_images(test_embedding, sample_embedding)
                        results.append((sample_file, similarity))
            
            # Sort results by similarity (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Display results
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Comparison results for: {os.path.basename(file_path)}\n\n")
            
            for sample, similarity in results:
                self.result_text.insert(tk.END, f"{sample}: {similarity*100:.2f}% similar\n")
            
            self.result_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparator(root)
    root.mainloop()