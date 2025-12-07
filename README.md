# ANPR-ATCC Streamlit Application  
### Automatic Number Plate Recognition & Traffic Classification System

## 📌 Project Overview

This project is a **Streamlit-based ANPR + ATCC system** that enables:

- **Automatic Number Plate Recognition (ANPR)**  
  Upload an image → detect the number plate → draw bounding box → extract OCR text using Tesseract.

- **Automatic Traffic Classification & Counting (ATCC)**  
  Upload a video → detect vehicle classes (car, bike, bus, truck, etc.) → count vehicles → show processing progress.

This system uses **YOLOv8**, **OpenCV**, and **Tesseract OCR** integrated into a simple and interactive **Streamlit web UI**, making it ideal for learning, academic submissions, demos, and real-time traffic analytics prototypes.

---

## ⭐ Key Features

### 🔹 ANPR (Image-Based Number Plate Recognition)
- Upload image input  
- YOLOv8 detects the number plate  
- Green bounding box drawn  
- Plate region cropped automatically  
- OCR text extracted using Tesseract  
- Recognized number displayed on the UI  
- High accuracy on Indian vehicle plates  

### 🔹 ATCC (Automatic Traffic Classification & Counting)
- Upload video input  
- YOLOv8 detects multiple vehicle classes  
- Counts car, bike, bus, truck, auto, etc.  
- Shows **percentage progress** of video processing  
- Outputs annotated video frames  
- Real-time Streamlit preview  

### 🔹 Streamlit UI Features
- Modern and clean interface  
- Two tabs: ANPR & ATCC  
- Real-time progress bar  
- Immediate results display  
- No need for command-line execution  

---

## 🛠 Technology Stack

| Component | Description |
|----------|-------------|
| **YOLOv8 (Ultralytics)** | For number plate & vehicle detection |
| **OpenCV** | Image and video processing |
| **Tesseract OCR** | Text extraction from number plate region |
| **Streamlit** | Interactive front-end interface |
| **Python 3.10+** | Core programming language |

---

## 📂 Project Structure

🧪 How the Application Works
🔸 ANPR Workflow

User uploads an image

YOLO model detects number plate

Green bounding box drawn

Plate cropped automatically

Tesseract OCR extracts text

Output text displayed on screen

🔸 ATCC Workflow

User uploads a traffic video

YOLO model detects vehicles frame-by-frame

Vehicle count maintained

Processing percentage updated live

Streamlit shows real-time annotated frame preview