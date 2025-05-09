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
    custom_model_path = "D:/Vietnamese-handwritten/data/final_model.keras"
    
    # Kiểm tra xem các file cần thiết có tồn tại không
    file_missing = False
    
    if not os.path.exists(char_to_idx_path):
        st.error(f"Không tìm thấy file từ điển ký tự tại: {char_to_idx_path}")
        file_missing = True
    
    if not os.path.exists(custom_model_path):
        st.error(f"Không tìm thấy file model tại: {custom_model_path}")
        file_missing = True
    
    if file_missing:
        st.stop()
    
    # Khởi tạo OCR
    @st.cache_resource
    def load_ocr_model():
        logger.info("Đang khởi tạo OCR model...")
        try:
            ocr = OCR(char_to_idx_path, custom_model_path=custom_model_path)
            return ocr
        except Exception as e:
            st.error(f"Lỗi khởi tạo OCR: {str(e)}")
            st.error(traceback.format_exc())
            return None
    
    with st.spinner("Đang tải mô hình OCR..."):
        ocr = load_ocr_model()
    
    if ocr is None:
        st.error("Không thể khởi tạo OCR model. Vui lòng kiểm tra logs để biết thêm chi tiết.")
        st.stop()
    
    st.success("OCR model đã được khởi tạo!")
    
    # Upload image
    uploaded_file = st.file_uploader("Upload ảnh chữ viết tay", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Lưu ảnh đã upload vào thư mục tạm
        image_path = os.path.join("temp", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Hiển thị ảnh đã upload
        st.image(image_path, caption="Ảnh đã tải lên")
        
        # Chọn phương pháp OCR
        use_vietocr = st.checkbox("Sử dụng VietOCR (pretrained)", value=True)
        
        # Debug information
        if not use_vietocr:
            if ocr.custom_model is None:
                st.warning("⚠️ Custom model chưa được tải. Không thể sử dụng chế độ custom model.")
            else:
                st.info("✓ Custom model đã được tải sẵn sàng sử dụng.")
        else:
            if ocr.vietocr_predictor is None:
                st.warning("⚠️ VietOCR model chưa được tải. Không thể sử dụng chế độ VietOCR.")
            else:
                st.info("✓ VietOCR model đã được tải sẵn sàng sử dụng.")
        
        # Nút nhận dạng
        if st.button("Nhận dạng chữ"):
            try:
                with st.spinner("Đang nhận dạng văn bản..."):
                    # Kiểm tra model được chọn đã được tải chưa
                    if (use_vietocr and ocr.vietocr_predictor is None):
                        st.error("VietOCR model chưa được tải. Vui lòng kiểm tra cài đặt VietOCR.")
                    elif (not use_vietocr and ocr.custom_model is None):
                        st.error("Custom model chưa được tải. Vui lòng kiểm tra đường dẫn đến model.")
                    else:
                        # Thực hiện nhận dạng
                        text = ocr.recognize(image_path, use_vietocr=use_vietocr)
                        
                        # Hiển thị kết quả
                        st.success("Nhận dạng thành công!")
                        st.write("Văn bản được nhận dạng:")
                        st.code(text)
                        
                        # Nút copy kết quả
                        if st.button("Copy văn bản"):
                            pyperclip.copy(text)
                            st.info("Đã copy văn bản vào clipboard!")
            except Exception as e:
                st.error(f"Lỗi khi nhận dạng: {str(e)}")
                st.error("Chi tiết lỗi:")
                st.code(traceback.format_exc())

except Exception as e:
    st.error(f"Lỗi không mong muốn: {str(e)}")
    st.error("Chi tiết lỗi:")
    st.code(traceback.format_exc())