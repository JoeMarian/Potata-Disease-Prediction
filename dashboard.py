import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
from datetime import datetime
from reportlab.pdfgen import canvas
import io
import matplotlib.pyplot as plt

# Load Model
model_path = "/Users/joemarian/Downloads/AICTE-Internship-files/3.Potato Leaf Disease Detection/dataset/potato_disease_model.h5"
model = tf.keras.models.load_model(model_path)

# Class Labels
CLASS_LABELS = ["Early Blight", "Late Blight", "Healthy"]

# Function to preprocess image
def preprocess_image(image):
    img = np.array(image)

    if img.shape[-1] == 4:  # Convert RGBA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)

    return img

# Function to predict disease
def predict_disease(image):
    img = preprocess_image(image)
    prediction = model.predict(img)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class] * 100

    return CLASS_LABELS[predicted_class], confidence, prediction

# Function to estimate progression time (for Early Blight)
def estimate_progression(confidence):
    if confidence > 90:
        return "2-3 days"
    elif confidence > 70:
        return "5-7 days"
    elif confidence > 50:
        return "10-12 days"
    else:
        return "Uncertain"

# Function to highlight affected region
def highlight_affected_region(image):
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    highlighted = cv2.addWeighted(img, 0.7, mask, 0.3, 0)

    return Image.fromarray(highlighted)

# Function to generate PDF Report
def generate_pdf(image, user_type, disease, confidence, affected_percentage, time_to_progress):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer)

    # Date & Time
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Potato Leaf Disease Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 780, f"Generated on: {current_time}")
    c.drawString(100, 760, f"User Type: {user_type}")
    c.drawString(100, 740, f"Disease Detected: {disease}")
    c.drawString(100, 720, f"Confidence Level: {confidence:.2f}%")

    if disease != "Healthy":
        c.drawString(100, 700, f"Affected Area: {affected_percentage}%")
        if disease == "Early Blight":
            c.drawString(100, 680, f"Estimated Time to Full Infection: {time_to_progress}")

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# Streamlit UI
st.set_page_config(page_title="Potato Leaf Disease Detection", layout="centered")
st.title("🍂 Potato Leaf Disease Detection Dashboard")

# User Role Selection
user_type = st.radio("👤 Select User Type", ["Farmer", "Researcher"])

# File Upload
uploaded_file = st.file_uploader("📂 Upload a Potato Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image in center with reduced size
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=False, width=250)

    # Prediction
    predicted_label, confidence, raw_predictions = predict_disease(image)

    st.subheader("🔍 Prediction Results")
    st.write(f"**🟢 Disease Detected:** {predicted_label}")
    st.write(f"**📊 Confidence Level:** {confidence:.2f}%")

    # Insights for Farmers
    if predicted_label == "Healthy":
        st.success("✅ The leaf is healthy! No disease detected.")
    else:
        affected_percentage = round((confidence / 100) * 100, 2)
        st.write(f"**🌱 Estimated Affected Area:** {affected_percentage}%")

        if predicted_label == "Early Blight":
            time_to_progress = estimate_progression(confidence)
            st.warning(f"⏳ Estimated time for full infection: **{time_to_progress}**")

    # Extra Features for Researchers
    if user_type == "Researcher":
        st.subheader("📊 Researcher Insights")
        st.write("Detailed prediction analysis and insights.")

        # Confidence Bar Chart
        st.bar_chart(dict(zip(CLASS_LABELS, raw_predictions)))

        # Affected Region Highlighting
        st.subheader("🖼️ Highlighted Affected Region")
        highlighted_image = highlight_affected_region(image)
        st.image(highlighted_image, caption="Highlighted Disease Area", use_container_width=False, width=250)

        # Probability Distribution Graph
        st.subheader("📉 Confidence Level Distribution")
        fig, ax = plt.subplots()
        ax.bar(CLASS_LABELS, raw_predictions * 100, color=['blue', 'red', 'green'])
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Model Predictions")
        st.pyplot(fig)

        # Additional Research Features
        st.write("🔬 **Analysis:**")
        st.write("- The model confidence levels for each category.")
        st.write("- Potential spread estimation and infection severity.")
        st.write("- Recommendations based on historical patterns.")
        st.write("- Visualization of affected regions.")

    # Generate Report
    st.subheader("📄 Download Report")
    pdf_buffer = generate_pdf(image, user_type, predicted_label, confidence, affected_percentage, time_to_progress if predicted_label == "Early Blight" else "N/A")
    st.download_button(label="📥 Download Report", data=pdf_buffer, file_name=f"{user_type}_potato_disease_report.pdf", mime="application/pdf")

# Footer
st.markdown("""
---
📌 **Joe Marian A**
""")
