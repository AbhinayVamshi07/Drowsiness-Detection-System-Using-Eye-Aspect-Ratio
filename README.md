# 😴 Drowsiness Detection System Using Eye Aspect Ratio (EAR)

A real-time **Driver Drowsiness Detection System** built using Python, OpenCV, and dlib.  
The system continuously monitors eye movement through a webcam and triggers an alert when prolonged eye closure is detected, helping to prevent accidents caused by driver fatigue.

---

## 🚗 Problem Statement
Drowsy driving is one of the major causes of road accidents globally.  
This project aims to **detect driver drowsiness in real time** by monitoring eye blink duration and alerting the user when they begin to fall asleep.

---

## 🚀 Features
- ✔️ Real-time face & eye detection using webcam  
- ✔️ Calculates **Eye Aspect Ratio (EAR)** for both eyes  
- ✔️ Detects prolonged eye closure (indicates sleepiness)  
- ✔️ Triggers **beep alert** when drowsiness is detected  
- ✔️ Displays EAR value on screen  
- ✔️ Draws green eye contours & red alert box during drowsiness  

---

## 🧠 How It Works
1️⃣ Detect face using **dlib frontal face detector**  
2️⃣ Extract **68 facial landmark points**  
3️⃣ Identify left & right eye regions  
4️⃣ Calculate **Eye Aspect Ratio (EAR)**  
5️⃣ If EAR stays below threshold for required time → **Drowsiness Detected**  
6️⃣ System beeps + shows alert message + highlights face

**EAR Formula**

\[
EAR = \frac{||p2 - p6|| + ||p3 - p5||}{2 \times ||p1 - p4||}
\]

---

## ⚙️ Constants Used
| Constant | Description |
|--------|-------------|
| `EYE_AR_THRESH = 0.25` | EAR threshold to detect closed eyes |
| `EYE_AR_CONSEC_FRAMES = 20` | Frames required to confirm drowsiness |

You can tune these values according to your environment.

---

## 🛠 Technologies Used
- Python
- OpenCV
- dlib
- imutils
- SciPy
- winsound (Windows alert sound)

---

## 📦 Installation

### 1️⃣ Install Required Libraries
```bash
pip install opencv-python
pip install dlib
pip install imutils
pip install scipy
