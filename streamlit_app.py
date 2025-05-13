import streamlit as st
import os
import sys
import traceback
import logging
import pyperclip

# Custom imports
module_path = os.path.join(os.path.dirname(__file__), "module")
if module_path not in sys.path:
    sys.path.append(module_path)

# Suppress logging
logging.getLogger().setLevel(logging.CRITICAL)

# Import OCR module
try:
    from ocr_app import OCR
except ImportError as e:
    st.error(f"Không thể import module OCR: {e}")
    st.stop()

# Paths configuration
CHAR_DICT_PATH = "D:/Vietnamese-handwritten/data/char_to_idx_simplified.json"
MODEL_PATH = "D:/Vietnamese-handwritten/data/best_model.keras"

# Ensure paths exist
if not os.path.exists(CHAR_DICT_PATH):
    st.error(f"Không tìm thấy từ điển ký tự: {CHAR_DICT_PATH}")
    st.stop()
if not os.path.exists(MODEL_PATH):
    st.error(f"Không tìm thấy mô hình: {MODEL_PATH}")
    st.stop()

# Create temp directory
os.makedirs("temp", exist_ok=True)

# Sidebar configuration
def configure_sidebar():
    # Centered logo with custom styling
    st.sidebar.markdown("""
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <img src="https://ft.ptithcm.edu.vn/wp-content/uploads/2021/08/PTIT-1170x1264.png" 
             style="max-width: 100px; height: auto;">
    </div>
    """, unsafe_allow_html=True)

    # Horizontal line
    st.sidebar.markdown("<hr style='border: 1px solid #4a4a4a; margin: 10px 0;'>", unsafe_allow_html=True)
    
    # Project Title
    st.sidebar.markdown("### Dự Án Nghiên Cứu")
    st.sidebar.markdown("### Ứng dụng nhận dạng chữ viết tay ")
    
    # Team Section with more visual appeal
    st.sidebar.markdown("## Nhóm Nghiên Cứu")
    team_members = [
        "Nguyễn Văn An - N22DCAT001",
        "Trần Xuân Đông - N22DCAT018",
        "Lê Đình Nghĩa - N22DCAT038",
        "Lê Trần Gia Thân - N22DCAT050",
    ]
    
    for member in team_members:
        st.sidebar.markdown(member)
    
    

# Cache OCR model
@st.cache_resource
def load_ocr():
    try:
        return OCR(CHAR_DICT_PATH, custom_model_path=MODEL_PATH)
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo OCR: {e}")
        return None

# Main Streamlit app
def main():
    # Configure page
    st.set_page_config(
        page_title="Nhận Dạng Chữ Viết Tay",
        page_icon="📝",
        layout="wide"
    )
    
    # Sidebar
    configure_sidebar()
    
    # Main content
    st.title("🖋️ Hệ Thống Nhận Dạng Chữ Viết Tay Tiếng Việt")
    st.markdown("""
    ### Giới Thiệu
    Ứng dụng tiên tiến sử dụng trí tuệ nhân tạo để nhận dạng chữ viết tay tiếng Việt.
    """)

    # Load OCR model
    ocr = load_ocr()
    if ocr is None:
        st.stop()

    # Recognition method selection (always visible)
    col_method = st.columns(1)[0]
    with col_method:
        use_vietocr = st.radio(
            "Chọn Phương Pháp Nhận Dạng", 
            ["VietOCR (Pretrained)", "Mô Hình Tùy Chỉnh"],
            index=0,
            horizontal=True
        )

    # Image upload section
    uploaded_file = st.file_uploader(
        "📤 Tải ảnh chữ viết tay lên", 
        type=["jpg", "jpeg", "png"],
        help="Vui lòng tải ảnh chữ viết tay có độ phân giải rõ nét"
    )

    if uploaded_file:
        # Save uploaded image
        image_path = os.path.join("temp", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display uploaded image and result in columns
        col_img, col_result = st.columns(2)
        
        with col_img:
            st.image(image_path, caption="📷 Ảnh Đã Tải Lên", use_container_width=True)
        
        with col_result:
            # Automatic recognition
            with st.spinner("Đang xử lý nhận dạng..."):
                try:
                    # Determine recognition method
                    use_vietocr_bool = use_vietocr == "VietOCR (Pretrained)"
                    
                    # Validate model availability
                    if use_vietocr_bool and not ocr.vietocr_predictor:
                        st.error("VietOCR chưa được tải.")
                    elif not use_vietocr_bool and not ocr.custom_model:
                        st.error("Mô hình tùy chỉnh chưa được tải.")
                    else:
                        # Perform recognition
                        result = ocr.recognize(image_path, use_vietocr=use_vietocr_bool)
                        
                        # Display results
                        st.success("✅ Nhận Dạng Thành Công!")
                        result_container = st.empty()
                        result_container.code(result, language="text")
                        
                        # Copy to clipboard option
                        if st.button("📋 Sao Chép Kết Quả"):
                            pyperclip.copy(result)
                            st.toast("Đã sao chép văn bản vào clipboard!")

                except Exception as e:
                    st.error(f"Lỗi khi nhận dạng: {e}")

# Run the app
if __name__ == "__main__":
    main()