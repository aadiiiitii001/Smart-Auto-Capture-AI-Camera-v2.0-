import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import matplotlib.pyplot as plt

# Streamlit page configuration
st.set_page_config(page_title="Smart Auto-Capture AI Camera", page_icon="📸", layout="wide")

# Title and description
st.title("📸 Smart Auto-Capture AI Camera v2.0")
st.markdown("""
Upload an image to detect **faces and emotions** using AI-powered vision models (Mediapipe + OpenCV).  
Now with **Emotion Analytics Dashboard** and **Detection History** 🧠
""")

# Initialize session state for emotion tracking
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

# Sidebar for dashboard
st.sidebar.header("📊 Emotion Dashboard")
if st.sidebar.button("Reset History"):
    st.session_state.emotion_history = []
    st.sidebar.success("History cleared!")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Simple smile-based emotion detector
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotion_label = "Neutral"
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            emotion_label = "Happy"
            break
    return emotion_label

if uploaded_file is not None:
    st.info("Processing image... please wait ⏳")
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Detect faces
    with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

    # Emotion detection
    emotion = detect_emotion(frame)
    st.session_state.emotion_history.append(emotion)

    filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)

    # Display processed image and result
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Detected Emotion: {emotion}", use_column_width=True)
    st.success(f"✅ Emotion Detected: {emotion}")
    st.download_button("📥 Download Processed Image", data=open(filename, "rb").read(), file_name=filename, mime="image/jpeg")

# Sidebar dashboard visualization
if st.session_state.emotion_history:
    st.sidebar.subheader("Emotion Summary")
    happy_count = st.session_state.emotion_history.count("Happy")
    neutral_count = st.session_state.emotion_history.count("Neutral")

    fig, ax = plt.subplots()
    ax.pie([happy_count, neutral_count], labels=["Happy 😀", "Neutral 😐"], autopct="%1.1f%%", startangle=90)
    st.sidebar.pyplot(fig)

    st.sidebar.write(f"🧮 Total Detections: {len(st.session_state.emotion_history)}")
    st.sidebar.write(f"😊 Happy: {happy_count}")
    st.sidebar.write(f"😐 Neutral: {neutral_count}")
