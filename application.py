import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
import random
import io
import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Function to load the pretrained models
def load_models():
    # Replace "my_model.h5" with the appropriate model paths for your infection and ischaemia models
    infection_model = tf.keras.models.load_model("my_model.h5")
    ischaemia_model = tf.keras.models.load_model("my_model.h5")
    return infection_model, ischaemia_model

# Function to preprocess the uploaded image
def preprocess_img(uploaded_file, target_size=(224, 224)):
    img = image.load_img(uploaded_file, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(img_array, axis=0)

# Function to generate a PDF report with three sections using ReportLab
def generate_pdf(patient_df, report_df, treatment_df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("Final Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Patient Information
    elements.append(Paragraph("Patient Information", styles['Heading2']))
    patient_data = [patient_df.columns.tolist()] + patient_df.values.tolist()
    patient_table = Table(patient_data, hAlign="LEFT")
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 12))
    
    # Report (Classification Results)
    elements.append(Paragraph("Report", styles['Heading2']))
    report_data = [report_df.columns.tolist()] + report_df.values.tolist()
    report_table = Table(report_data, hAlign="LEFT")
    report_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    elements.append(report_table)
    elements.append(Spacer(1, 12))
    
    # Suggestive Treatment
    elements.append(Paragraph("Suggestive Treatment", styles['Heading2']))
    treatment_data = [treatment_df.columns.tolist()] + treatment_df.values.tolist()
    treatment_table = Table(treatment_data, hAlign="LEFT")
    treatment_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    elements.append(treatment_table)
    
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

def main():
    st.title("Wound Classification with AI")
    
    st.write("""
    This application uses artificial intelligence to classify wounds.
    It analyzes an uploaded wound image using two models:
    one for detecting infection and another for detecting ischaemia.
    Please enter your details on the sidebar, upload an image, and then view your results.
    """)

    # Sidebar for personal information
    st.sidebar.header("Enter Your Information")
    name = st.sidebar.text_input("Name")
    user_id = st.sidebar.text_input("ID")
    address = st.sidebar.text_input("Address")
    phone = st.sidebar.text_input("Phone Number")

    # Instructions for the user
    st.write("### Upload Wound Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image at a reduced size (e.g., width=200 pixels)
        st.image(uploaded_file, caption="Uploaded Image", width=200)
        
        # Preprocess the image
        img = preprocess_img(uploaded_file, target_size=(224, 224))
        
        # Load the two models
        with st.spinner("Loading models..."):
            infection_model, ischaemia_model = load_models()
        
        # Predict using both models
        infection_pred = infection_model.predict(img)
        ischaemia_pred = ischaemia_model.predict(img)
        
        # Convert predictions into human-readable results (threshold 0.5)
        infection_result = "Positive" if infection_pred[0][0] > 0.5 else "Negative"
        ischaemia_result = "Positive" if ischaemia_pred[0][0] > 0.5 else "Negative"
        
        # Create DataFrames for each section
        patient_df = pd.DataFrame({
            "Name": [name],
            "ID": [user_id],
            "Address": [address],
            "Phone": [phone]
        })
        
        report_df = pd.DataFrame({
            "Infection": [infection_result],
            "Ischaemia": [ischaemia_result]
        })
        
        # Determine automated treatment recommendations based on classification results
        if infection_result == "Positive" and ischaemia_result == "Positive":
            dressing = "Foam, Alginate, Hydrofiber, Polymeric membrane"
            antibiotic = "May or may not be required (based on underlying cause)"
            surgical = "Find underlying cause; treat if necessary"
        elif infection_result == "Positive" and ischaemia_result == "Negative":
            dressing = "Standard dressing (no specific recommendation provided)"
            antibiotic = "Not specified"
            surgical = "Not specified"
        elif infection_result == "Negative" and ischaemia_result == "Positive":
            dressing = "Tulle, Hydrogel, Hydrocolloid, Silver dressing"
            antibiotic = "Yes"
            surgical = "Debridement may be needed"
        elif infection_result == "Negative" and ischaemia_result == "Negative":
            dressing = "All dressing materials except silver, charcoal, and advanced dressings"
            antibiotic = "Yes"
            surgical = ("Ready for secondary wound closure; if wound is small, continue dressing until the wound heals")
        else:
            dressing = "Not specified"
            antibiotic = "Not specified"
            surgical = "Not specified"
        
        treatment_df = pd.DataFrame({
            "Dressing Material": [dressing],
            "Antibiotic": [antibiotic],
            "Surgical Procedure": [surgical]
        })
        
        # Display the three tables separately
        st.write("### Patient Information")
        st.dataframe(patient_df, use_container_width=True)
        
        st.write("### Report")
        st.dataframe(report_df, use_container_width=True)
        
        st.write("### Suggestive Treatment")
        st.dataframe(treatment_df, use_container_width=True)
        
        # Generate PDF report and provide a download button
        # File name: patientID_currentDate.pdf (e.g., 11191029023_20250204.pdf)
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        file_name = f"{user_id}_{current_date}.pdf"
        pdf_bytes = generate_pdf(patient_df, report_df, treatment_df)
        st.download_button(
            label="Download Final Report as PDF",
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
