import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import pytesseract
from scipy.spatial import distance as dist
from collections import defaultdict

# -------------------- TESSERACT CONFIG --------------------
# Set correct Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------- LOAD YOLO MODEL --------------------
@st.cache_resource
def load_model():
    return YOLO("ANPR/weights/best.pt")   # single model for ANPR + ATCC

model = load_model()

# -------------------- ANPR FUNCTIONS --------------------
def extract_plate_text(plate_roi):
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, config="--psm 7")
    return text.strip()

def perform_anpr(image):
    results = model.predict(image)
    annotated = image.copy()
    final_text = ""

    for res in results:
        for box in res.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate_roi = image[y1:y2, x1:x2]

            plate_text = extract_plate_text(plate_roi)
            final_text = plate_text

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated, final_text

# -------------------- CENTROID TRACKER --------------------
class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = {}
        self.max_distance = max_distance

    def update(self, detections):
        if len(detections) == 0:
            return self.objects

        detections = np.array(detections)
        objects_new = {}

        if len(self.objects) == 0:
            for det in detections:
                objects_new[self.next_id] = det
                self.next_id += 1
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            if len(object_centroids.shape) != 2:
                object_centroids = object_centroids.reshape(-1, 2)

            D = dist.cdist(object_centroids, detections)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            assigned = set()

            for row, col in zip(rows, cols):
                if D[row, col] > self.max_distance:
                    continue
                obj_id = object_ids[row]
                objects_new[obj_id] = detections[col]
                assigned.add(col)

            for i, det in enumerate(detections):
                if i not in assigned:
                    objects_new[self.next_id] = det
                    self.next_id += 1

        self.objects = objects_new
        return self.objects

# -------------------- ATCC VIDEO PROCESSING --------------------
def process_atcc_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(temp_out.name, fourcc, fps, (w, h))

    tracker = CentroidTracker()
    vehicle_counts = defaultdict(int)
    progress = st.progress(0)
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        detections = []

        for res in results:
            for box in res.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append([cx, cy, x1, y1, x2, y2])

        tracked = tracker.update([d[:2] for d in detections])

        for obj_id, centroid in tracked.items():
            for det in detections:
                cx, cy, x1, y1, x2, y2 = det
                if abs(cx - centroid[0]) < 5 and abs(cy - centroid[1]) < 5:
                    vehicle_counts["Vehicle"] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    crop = frame[y1:y2, x1:x2]
                    text = extract_plate_text(crop)

                    if text != "":
                        cv2.putText(frame, text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)

        out.write(frame)
        frame_no += 1
        progress.progress(frame_no / total)

    cap.release()
    out.release()

    return temp_out.name, vehicle_counts

# -------------------- STREAMLIT UI --------------------
def main():
    st.title("🔍 ANPR & ATCC System")

    menu = ["ANPR (Image)", "ATCC (Video)"]
    choice = st.sidebar.selectbox("Choose Module", menu)

    if choice == "ANPR (Image)":
        st.subheader("Upload Image for Number Plate Recognition")
        file = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"])

        if file:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8),
                                 cv2.IMREAD_COLOR)
            annotated, text = perform_anpr(image)

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Detected Number Plate")

            st.success(f"Extracted Text: **{text}**")

    else:
        st.subheader("Upload Video for ATCC Processing")
        file = st.file_uploader("Choose Video", type=["mp4", "avi", "mov"])

        if file:
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(file.read())

            processed_path, counts = process_atcc_video(temp.name)

            st.video(processed_path)

            st.success("Vehicle Counts:")
            for k, v in counts.items():
                st.write(f"{k}: {v}")

            st.download_button(
                "Download Processed Video",
                data=open(processed_path, "rb"),
                file_name="ATCC_output.mp4",
                mime="video/mp4"
            )

if __name__ == "__main__":
    main()
