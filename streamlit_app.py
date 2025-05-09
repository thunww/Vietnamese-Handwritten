import streamlit as st
import cv2
import os
from ocr_app import OCR
import pyperclip
import traceback

try:
    # Initialize OCR with full characters and best model
    characters = '0123456789abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ .,:!?()"-/\''
    char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
    custom_model_path = "D:/Vietnamese-handwritten/data/final_model.keras"  # Use best model

    st.title("Vietnamese Handwritten Recognition")

    st.write("Initializing OCR model...")
    ocr = OCR(char_to_idx, custom_model_path=custom_model_path)
    st.success("OCR model initialized!")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Save uploaded image to temp directory
        os.makedirs("temp", exist_ok=True)
        image_path = os.path.join("temp", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display uploaded image
        st.image(image_path, caption="Uploaded Image")

        # Choose OCR method
        use_vietocr = st.checkbox("Use VietOCR (pretrained)", value=True)

        # Run OCR
        st.write("Recognizing text...")
        
        result = ocr.recognize(image_path, use_vietocr=use_vietocr)
        text = str(result).strip()    
        st.success("Recognized Text:")
        st.code(text, language="text")  # Hiển thị khối văn bản

        

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.error("Traceback:")
    st.code(traceback.format_exc())
