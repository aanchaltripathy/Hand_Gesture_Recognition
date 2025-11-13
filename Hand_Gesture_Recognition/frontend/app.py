import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import load_model
# from keras.layers import TFSMLayer
from keras.models import load_model
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")

st.title("🖐️ Hand Gesture Recognition App")

col1, col2 = st.columns(2)

with col1:
    st.header("Live Feed")
    FRAME_WINDOW = st.image([])
    
with col2:
    st.header("Detected Gesture")
    label_placeholder = st.empty()

# Load model and class names
# model = TFSMLayer('../mp_hand_gesture', call_endpoint='serving_default')
# model = tf.saved_model.load('../mp_hand_gesture')
# model = load_model('../mp_hand_gesture')

model = tf.saved_model.load('../mp_hand_gesture')
infer = model.signatures["serving_default"]
# print("Output keys:", infer.output_keys)
try:
    output_key = list(infer.structured_outputs.keys())[0]
    print("✅ Model loaded successfully.")
    print("Output key:", output_key)
except Exception as e:
    print("⚠️ Could not determine output key automatically:", e)
    print("Structured outputs:", infer.structured_outputs)
    output_key = list(infer.structured_outputs.keys())[0]
print(infer.structured_outputs)
f = open('../gesture.names', 'r')
classNames = f.read().split('\n')
f.close()

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        st.error("Failed to access webcam.")
        break

    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    className = ''

    if result.multi_hand_landmarks:
        h, w, c = frame.shape
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                landmarks.append([lm.x * w, lm.y * h])  # convert from 0–1 → pixel coordinates

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        # prediction = model(tf.convert_to_tensor([landmarks], dtype=tf.float32))
        # input_tensor = tf.convert_to_tensor([landmarks], dtype=tf.float32)
        # prediction = model.predict(np.array([landmarks], dtype=np.float32))
        # input_tensor = np.array([landmarks], dtype=np.float32)
        # prediction = model(input_tensor) 
        # input_tensor = tf.convert_to_tensor([landmarks], dtype=tf.float32)
        # input_tensor = tf.convert_to_tensor([np.array(landmarks).flatten()], dtype=tf.float32)
        # prediction = infer(input_tensor)['dense_2'].numpy() 
        # classID = np.argmax(prediction)
        # className = classNames[classID]
        landmarks = np.array(landmarks)

        # ✅ Step 1: Normalize positions relative to hand
        x_vals = landmarks[:, 0]
        y_vals = landmarks[:, 1]
        landmarks[:, 0] = x_vals - np.min(x_vals)
        landmarks[:, 1] = y_vals - np.min(y_vals)

        # ✅ Step 2: Scale to consistent range
        landmarks = landmarks / np.max(landmarks)

        # ✅ Step 3: Flatten and infer
        input_tensor = tf.convert_to_tensor([landmarks.flatten()], dtype=tf.float32)
        prediction = infer(input_tensor)['dense_2'].numpy()

        classID = np.argmax(prediction)
        className = classNames[classID]

#--------------------------------------------------


        # prediction = infer(input_tensor)['dense_2'].numpy()
        # classID = np.argmax(prediction)
        # className = classNames[classID]

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    label_placeholder.write(f"**Detected:** {className}")

cap.release()
