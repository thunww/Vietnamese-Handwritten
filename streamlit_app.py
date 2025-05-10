# streamlit_app.py
import streamlit as st
import os
import sys
import traceback
import logging
import pyperclip

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thêm module OCR vào path
module_path = os.path.join(os.path.dirname(__file__), "module")
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    from ocr_app import OCR
except ImportError as e:
    st.error(f"Không thể import module OCR: {e}")
    st.stop()

# Thiết lập đường dẫn tới model và từ điển
CHAR_DICT_PATH = "D:/Vietnamese-handwritten/data/char_to_idx_simplified.json"
MODEL_PATH = "D:/Vietnamese-handwritten/data/new_model.keras"

# Kiểm tra file tồn tại
if not os.path.exists(CHAR_DICT_PATH):
    st.error(f"Không tìm thấy từ điển ký tự: {CHAR_DICT_PATH}")
    st.stop()
if not os.path.exists(MODEL_PATH):
    st.error(f"Không tìm thấy mô hình: {MODEL_PATH}")
    st.stop()

# Tạo thư mục tạm
os.makedirs("temp", exist_ok=True)

# Cache OCR model
@st.cache_resource
def load_ocr():
    try:
        logger.info("Đang tải mô hình OCR...")
        return OCR(CHAR_DICT_PATH, custom_model_path=MODEL_PATH)
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo OCR: {e}")
        st.code(traceback.format_exc())
        return None

# UI chính
st.title("📝 Nhận dạng chữ viết tay tiếng Việt")

ocr = load_ocr()
if ocr is None:
    st.stop()

st.success("✅ Mô hình đã được tải thành công!")

uploaded_file = st.file_uploader("📤 Tải ảnh viết tay lên", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image_path = os.path.join("temp", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="📷 Ảnh đã tải lên", use_column_width=True)

    # Chọn phương pháp nhận dạng
    use_vietocr = st.checkbox("Sử dụng VietOCR (pretrained)", value=True)

    if use_vietocr:
        if ocr.vietocr_predictor:
            st.info("VietOCR đã sẵn sàng.")
        else:
            st.warning("⚠️ Chưa thể sử dụng VietOCR.")
    else:
        if ocr.custom_model:
            st.info("Mô hình tùy chỉnh đã sẵn sàng.")
        else:
            st.warning("⚠️ Mô hình tùy chỉnh chưa được tải.")

    if st.button("🔍 Nhận dạng chữ"):
        with st.spinner("Đang xử lý..."):
            try:
                if use_vietocr and not ocr.vietocr_predictor:
                    st.error("VietOCR chưa được tải.")
                elif not use_vietocr and not ocr.custom_model:
                    st.error("Mô hình tùy chỉnh chưa được tải.")
                else:
                    result = ocr.recognize(image_path, use_vietocr=use_vietocr)
                    st.success("✅ Nhận dạng thành công!")
                    st.code(result)

                    if st.button("📋 Copy kết quả"):
                        pyperclip.copy(result)
                        st.info("Đã copy văn bản vào clipboard.")
            except Exception as e:
                st.error(f"Lỗi khi nhận dạng: {e}")
                st.code(traceback.format_exc())
