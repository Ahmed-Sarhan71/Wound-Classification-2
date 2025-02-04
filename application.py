import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Function to load the pretrained models
def load_models():
    infection_model = tf.keras.models.load_model("my_model.h5")
    ischaemia_model = tf.keras.models.load_model("my_model.h5")
    return infection_model, ischaemia_model

# Function to preprocess the uploaded image
def preprocess_img(uploaded_file, target_size=(224, 224)):
    img = image.load_img(uploaded_file, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image pixels to [0, 1]
    return np.expand_dims(img_array, axis=0)

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
        infection_result = "Positive" if infection_pred[0][0] < 0.5 else "Negative"
        ischaemia_result = "Positive" if ischaemia_pred[0][0] < 0.5 else "Negative"
        
        # Display classification results in Table 1
        st.write("### Classification Results (Table 1)")
        results_df = pd.DataFrame({
            "Infection": [infection_result],
            "Ischaemia": [ischaemia_result]
        })
        st.table(results_df)
        
        # Treatment recommendation inputs (Table 2)
        st.write("### Treatment Recommendations (Table 2)")
        dressing = st.text_input("Dressing Material", "")
        antibiotic = st.text_input("Antibiotic", "")
        surgical = st.text_input("Surgical Procedure", "")
        
        # Generate final report on button click
        if st.button("Generate Final Report"):
            report_df = pd.DataFrame({
                "Name": [name],
                "ID": [user_id],
                "Address": [address],
                "Phone": [phone],
                "Infection": [infection_result],
                "Ischaemia": [ischaemia_result],
                "Dressing Material": [dressing],
                "Antibiotic": [antibiotic],
                "Surgical Procedure": [surgical]
            })
            st.write("### Final Report")
            st.table(report_df)

if __name__ == "__main__":
    main()
