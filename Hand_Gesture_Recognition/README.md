# 🖐️ Hand Gesture Recognition

## 📘 Overview
This project aims to develop an **AI-based gesture recognition system** that can detect and classify static hand gestures in real time using a standard webcam.  
It leverages **MediaPipe Hands** for hand landmark detection and a **custom-trained Convolutional Neural Network (CNN)** for gesture classification.

### 💡 Use Case Example:
This system can serve as the foundation for:
- Gesture-controlled smart home interfaces  
- Robotics control via hand movements  
- Accessibility tools for differently abled individuals  
- Touchless control in public kiosks or AR/VR systems  

---

## 🌟 Features
✅ Real-time hand tracking and gesture recognition  
✅ Integration with **MediaPipe** for accurate 3D hand landmark extraction  
✅ Lightweight model (TensorFlow Lite compatible)  
✅ Deployed using **Streamlit** with a live webcam feed  
✅ Extendable for **dynamic gestures** (future work)

---

## ⚙️ System Architecture
Camera Input → MediaPipe Hand Landmarks → Feature Extraction → CNN Model → Gesture Classification → Action Mapping

---

## 📊 Dataset
- **Source:** Custom dataset created from webcam captures  
- **Gestures:** Peace ✌️, Thumbs Up 👍, Stop ✋, etc
- **Preprocessing:**  
  - Extracted 21 landmarks using MediaPipe  
  - Normalized (x, y) coordinates  
  - Augmentations: rotation, flipping, scaling  
- **Split:** 80% training, 20% validation  
- **Format:** Stored in `data.pickle` and labeled using `gesture.names`

---

## 🧠 Model and Training
- **Model Type:** CNN trained on 2D landmark coordinates  
- **Framework:** TensorFlow / Keras  
- **Optimizer:** Adam (lr=0.001)  
- **Loss Function:** Categorical Crossentropy  
- **Metrics:** Accuracy  

| Metric | Value |
|:-------|:-------:|
| Validation Accuracy | ~90% |


📈 *Accuracy/loss curves*
![Accuracy Curves](demo/screenshots/Screenshot%202025-11-13%20at%204.10.36%E2%80%AFPM.png)---

## 🧾 Results
- Model correctly classifies static gestures in real time  
- Average inference latency: **<50ms per frame** on Mac M1/M2  
- Smooth frame rate and minimal flickering  



📊 *Confusion matrix*
![Confusion Matrix](demo/screenshots/Screenshot%202025-11-13%20at%204.12.30%E2%80%AFPM.png)

---
## 📰 Read My Blog

### Curious about how I trained, tested, and debugged the model?
#### Check out my full write-up on Medium:

👉 [My First Try at Real-Time Hand Gesture Recognition — Lessons, Bugs, and How I Fixed Them](https://medium.com/@aanchaltripathy24/my-first-try-at-real-time-hand-gesture-recognition-lessons-bugs-and-how-i-fixed-them-fef059f7ffb2)



---
## 🎥 Demo
🎬 Watch the full demo here: [**YouTube Demo (https://youtu.be/veyhzNH6hwc)**]  
Or run it locally following the steps below 👇  

---

## 🧩 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/aanchaltripathy/Hand_Gesture_Recognition.git
cd Hand_Gesture_Recognition
