"""
Module để sử dụng thư viện vietocr cho nhận dạng chữ viết tay tiếng Việt
"""
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import os
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable để lưu model sau khi load
_vietocr_model = None

def load_vietocr_model(model_name='vgg_transformer', device='cpu', use_gpu=False):
    """
    Tải model VietOCR từ thư viện
    
    Args:
        model_name (str): Tên model ('vgg_transformer', 'vgg_seq2seq', 'resnet_transformer')
        device (str): Device để chạy model ('cpu' hoặc 'cuda:0')
        use_gpu (bool): Sử dụng GPU nếu có
    
    Returns:
        Predictor: Model VietOCR đã được tải
    """
    global _vietocr_model
    
    # Nếu model đã được tải, trả về model đó
    if _vietocr_model is not None:
        return _vietocr_model
    
    try:
        # Thiết lập cấu hình
        config = Cfg.load_config_from_name(model_name)
        
        # Nếu use_gpu=True và có GPU, sử dụng GPU
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda:0'
                    logger.info("Sử dụng GPU để xử lý")
                else:
                    logger.info("Không tìm thấy GPU, sử dụng CPU")
            except ImportError:
                logger.warning("Không thể import torch để kiểm tra GPU, sử dụng CPU")
        
        # Thiết lập device
        config['device'] = device
        
        # Các cấu hình bổ sung
        config['predictor']['beamsearch'] = False  # Tắt beamsearch để tốc độ nhanh hơn
        
        # Tải model
        logger.info(f"Đang tải model {model_name}...")
        predictor = Predictor(config)
        
        # Lưu model vào biến global
        _vietocr_model = predictor
        logger.info("Đã tải xong model VietOCR")
        
        return predictor
        
    except Exception as e:
        logger.error(f"Lỗi khi tải model VietOCR: {str(e)}")
        raise

def recognize_text_vietocr(image_path, predictor=None):
    """
    Nhận diện văn bản từ ảnh sử dụng VietOCR
    
    Args:
        image_path (str): Đường dẫn đến file ảnh
        predictor (Predictor, optional): Model VietOCR đã được tải. Nếu None, model sẽ được tải tự động.
    
    Returns:
        str: Văn bản được nhận diện
    """
    try:
        # Kiểm tra file ảnh
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh tại {image_path}")
        
        # Nếu predictor chưa được cung cấp, tải model
        if predictor is None:
            predictor = load_vietocr_model()
        
        # Đọc ảnh
        img = Image.open(image_path).convert('RGB')
        
        # Dự đoán
        logger.info(f"Đang nhận diện văn bản trong ảnh {os.path.basename(image_path)}...")
        text = predictor.predict(img)
        logger.info(f"Kết quả nhận diện: {text}")
        
        return text
        
    except Exception as e:
        logger.error(f"Lỗi khi nhận diện văn bản: {str(e)}")
        raise

if __name__ == "__main__":
    # Đọc đường dẫn ảnh từ người dùng
    image_path = input("Nhập đường dẫn đến file ảnh: ")
    
    try:
        # Tải model
        predictor = load_vietocr_model(use_gpu=True)
        
        # Nhận diện văn bản
        text = recognize_text_vietocr(image_path, predictor)
        
        print(f"Văn bản được nhận diện: {text}")
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")