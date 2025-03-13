import streamlit as st
import torch
import cv2
import pytesseract
import tempfile
import os
from PIL import Image
import numpy as np

# Load the trained YOLO model (replace 'best.pt' with your actual model path)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

def process_image(input_image):
    # Convert the uploaded image to an OpenCV format
    image = np.array(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Use YOLO model to detect license plates
    results = model(image)
    detected_boxes = results.xyxy[0]  # Bounding boxes, confidence scores, and class IDs

    # Loop through all the detected bounding boxes
    for box in detected_boxes:
        x1, y1, x2, y2, conf, cls = map(int, box[:6])  # Extract bounding box coordinates and confidence
        if conf > 0.5:  # You can adjust the confidence threshold as needed
            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Optionally, draw the confidence score and label (use class names for 3 classes)
            if cls == 0:
                label = "Analog License Plate"
            elif cls == 1:
                label = "Digital License Plate"
            elif cls == 2:
                label = "Non-License Plate"
            else:
                label = "Unknown"
            
            # Draw label and confidence on image
            cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Optionally, collect the bounding box coordinates for further processing (e.g., OCR)
            license_plate = image[y1:y2, x1:x2]
            # Convert to grayscale for better OCR results
            gray_license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

            # Use Tesseract OCR to extract text from Bangla license plates (adjust config as needed)
            text = pytesseract.image_to_string(gray_license_plate, config="--psm 6 -l ben")  # 'ben' is for Bangla
            print(f"Detected License Plate Text: {text.strip()}")

    # Convert the image back to RGB for display in Streamlit
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def main():
    st.title("License Plate Detection with YOLO and OCR")

    st.write("Upload an image file for license plate detection and OCR processing.")

    # Upload image file
    input_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if input_image is not None:
        # Open the uploaded image using PIL
        image = Image.open(input_image)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image
        st.write("Processing image... This may take some time.")
        processed_image = process_image(image)

        # Display the processed image
        st.image(processed_image, caption="Processed Image", use_column_width=True)

        # Provide download link for processed image
        st.download_button(
            label="Download Processed Image",
            data=processed_image.tobytes(),
            file_name="output_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
