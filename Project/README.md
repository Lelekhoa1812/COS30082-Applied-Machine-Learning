# 🧠 Face Recognition Attendance System

> **A real-time, anti-spoofing facial attendance system** powered by CNN, CascadeClassifier, DeepFace, and Silent-Face CNN ensemble.

---

## 🎯 Features

| Component            | Method / Library                                                                                           |
| -------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Face Embedding & Comparision**   | CNN Classifier and Triplet Loss                                                     |                                |
| **Face Detection Bbox**    |      CascadeClassifier                                                                                     |
| **Face Matching**    | Cosine similarity                                                                                          |
| **Emotion Analysis** | [DeepFace](https://github.com/serengil/deepface)                                                           |
| **Anti-Spoofing**    | [Silent-Face Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (3-model ensemble) |
| **Interface**        | Tkinter + ttk                                                                                              |

---

## 🧩 System Architecture

```
+----------------------+        +-------------------+        +-------------------+
|   Webcam Capture     | ---->  | CascadeClassifier | ---->  | Face Crop & Align |
+----------------------+        +-------------------+        +-------------------+
                                                         |
                                                         v
                                         +----------------------------+
                                         | Classifier or              |
                                         |  Triplet Loss on CNN       |
                                         +----------------------------+
                                                  |
       +------------------------------+           |         +---------------------+
       | Compare w/ known embeddings  | <---------+--------> | Cosine Similarity  |
       +------------------------------+                     +---------------------+
                |
         +-------------+
         | Match Label |
         +-------------+

Additionally:
- Cropped face passed to **DeepFace** for emotion.
- Same crop passed to **Silent-Face** for spoof detection.
```

---

## 💻 Demo UI

The app opens a GUI window with:

* Live video feed
* Real-time face recognition with name and emotion
* Anti-spoofing status (Live/Spoof)
* Snapshot saving
* Name prompt for unknown users

---

## 🚀 Installation & Setup

### 1. Create and activate virtual environment

```bash
# Linux / macOS
python3 -m venv cos30082-env
source cos30082-env/bin/activate

# Windows (Command Prompt)
python -m venv cos30082-env
cos30082-env\Scripts\activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download or prepare required models

| Model                | Source                                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------------------- |
| `mobilefacenet.onnx` | [Download](https://github.com/deepinsight/insightface) or from provided link                                  |
| `anti_spoof_models`  | Copied from [Silent-Face Repo](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) under `resources/` |

Ensure the following structure:

```
your_project/
├── src/                       # from Silent-Face repo
├── resources/
│   └── anti_spoof_models/
├── known_faces/
├── snapshot/
├── project.py
├── spoof_detector.py
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

* Python 3.7–3.10
* Tested on macOS & Windows
* Dependencies include:

  * `opencv-python`
  * `mediapipe`
  * `onnxruntime`
  * `deepface`
  * `numpy`
  * `tkinter` (preinstalled on most systems)
  * `torch`, `torchvision` for Silent-Face models

---

## 📸 Face Registration & Snapshots

* **Auto-prompt**: Unknown faces trigger a GUI popup to assign name.
* **Snapshot**: System saves one snapshot per recognized user per session.
* All saved images are stored in the `snapshot/` folder.

---

## 🔒 Anti-Spoofing System

**Silent-Face Ensemble**:

* Combines 3 lightweight CNNs for robust liveness detection.
* Runs on CPU in real-time.
* Prevents spoof attacks using printed photos or screen replays.

---

## 🎭 Emotion Detection

Powered by **DeepFace**, the system classifies emotions such as:

* Happy
* Sad
* Neutral
* Angry
* Surprise
* Fear
* Disgust

---

## 🧠 Future Work

* Improve recognition robustness using better augmentation
* Support for profile or angled face matching
* Export attendance logs (CSV)
* Add auto-tracking via servo (for physical attendance kiosks)

---

## 👤 Author

**Dang Khoa Le**
📅 Project Date: May 20, 2025
🎓 Bachelor of Software Engineering — Swinburne University

---

## 📎 References

* [MediaPipe](https://github.com/google/mediapipe)
* [InsightFace/MobileFaceNet](https://github.com/deepinsight/insightface)
* [Silent-Face Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
* [DeepFace](https://github.com/serengil/deepface)
