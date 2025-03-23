import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load YOLO model
@st.cache_resource
def load_model():
    # Load a YOLO model trained for bone fracture detection (update with your model path)
    model = YOLO("C:/Users/kasot/Desktop/New bone yolo/best.pt")  # Update with your specific model path
    return model

# Function to perform bone fracture detection
def detect_bone_fractures(image, model):
    # Convert image to numpy array
    image_np = np.array(image)

    # Perform detection
    results = model(image_np)
    detections = results[0].boxes.data.cpu().numpy()  # Get detection boxes and classes

    # Assuming your model has specific class indices for bone fractures; adjust as necessary
    fracture_class_indices = [0, 1, 2]  # Update this based on your model's class indices for bone fractures
    
    # Separate detections into fractures and non-fractures
    filtered_detections = [det for det in detections if int(det[5]) in fracture_class_indices]
    non_fracture_detections = [det for det in detections if int(det[5]) not in fracture_class_indices]

    # Annotate image with filtered detections (fractures)
    for det in filtered_detections:
        x1, y1, x2, y2, confidence, _ = det
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image_np, f"Fracture {confidence:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Annotate image with non-fracture detections (if needed)
    for det in non_fracture_detections:
        x1, y1, x2, y2, confidence, _ = det
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image_np, f"Non-Fracture {confidence:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image_np, filtered_detections + non_fracture_detections

# Streamlit UI
st.title("Bone Fracture Detection Application Using YOLO")

st.text("Upload an image to detect bone fractures")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image and resize it
    image = Image.open(uploaded_file).convert("RGB")
    
    # Resize the image to a fixed size (e.g., 640x640) for consistent input to the model
    image = image.resize((340,340))
    
    st.image(image, caption="Image Before Prediction", use_column_width=True)
# Load model
    model = load_model()

    # Perform detection
    with st.spinner("Detecting bone fractures..."):
        annotated_image, detections = detect_bone_fractures(image, model)

    # Display results
    st.image(annotated_image, caption="Detected Bone Fractures", use_column_width=True)

    # Display detection details
    st.subheader("Detection Details")
    
    if len(detections) == 0:
        st.write("No bone fractures detected.")
    else:
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, confidence, _ = detection
            if int(detection[5]) in [0, 1, 2]:  # Fracture class indices
                st.write(f"Fracture {i + 1}:")
            else:  # Non-fracture class indices
                st.write(f"Non-Fracture {i + 1}:")
            st.write(f"  - Confidence: {confidence:.2f}")
            st.write(f"  - Bounding Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")


st.text("Developed by SOLUTION SEEKARS(SHUBHAM,NIRANJAN,ABHISHEK,RAHUL)")
st.text("Guided by Prof A.P.PATIL MAM")
            
    