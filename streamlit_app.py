import streamlit as st
import cv2
import os
import sys
import json
import pyperclip
import traceback
from ocr_app import OCR
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thêm đường dẫn đến thư mục chứa module cần thiết
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "module")
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    # Đảm bảo rằng các thư mục tạm thời tồn tại
    os.makedirs("temp", exist_ok=True)
    
    st.title("Vietnamese Handwritten Recognition")
    
    # Đường dẫn đến các file cần thiết
    char_to_idx_path = "D:/Vietnamese-handwritten/data/char_to_idx.json"
    
    # Khởi tạo OCR chỉ với VietOCR (không cần model tùy chỉnh)
    @st.cache_resource
    def load_ocr_model():
        logger.info("Đang khởi tạo VietOCR model...")
        try:
            ocr = OCR(char_to_idx_path)  # Không truyền custom_model_path
            return ocr
        except Exception as e:
            st.error(f"Lỗi khởi tạo OCR: {str(e)}")
            st.error(traceback.format_exc())
            return None

    with st.spinner("Đang tải mô hình VietOCR..."):
        ocr = load_ocr_model()
    
    if ocr is None or ocr.vietocr_predictor is None:
        st.error("Không thể khởi tạo mô hình VietOCR. Vui lòng kiểm tra lại.")
        st.stop()

    st.success("✓ Mô hình VietOCR đã được tải!")

    # Upload ảnh
    uploaded_file = st.file_uploader("Tải ảnh chữ viết tay", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Lưu ảnh vào thư mục tạm
        image_path = os.path.join("temp", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(image_path, caption="Ảnh đã tải lên")
        
        # Nhận dạng
        if st.button("Nhận dạng chữ"):
            try:
                with st.spinner("Đang nhận dạng văn bản..."):
                    text = ocr.recognize(image_path, use_vietocr=True)

                    st.success("✓ Nhận dạng thành công!")
                    st.write("Văn bản được nhận dạng:")
                    st.code(text)

                    if st.button("Copy văn bản"):
                        pyperclip.copy(text)
                        st.info("Đã copy văn bản vào clipboard!")
            except Exception as e:
                st.error(f"Lỗi khi nhận dạng: {str(e)}")
                st.code(traceback.format_exc())

except Exception as e:
    st.error(f"Lỗi không mong muốn: {str(e)}")
    st.code(traceback.format_exc())
