import logging
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_vietocr_model(use_gpu=False):
    """
    Tải mô hình VietOCR với cấu hình tối ưu cho chữ viết tay.
    """
    logger.info("Đang tải cấu hình VietOCR...")
    try:
        config = Cfg.load_config_from_name('vgg_transformer')  # Thử vgg_transformer
        logger.info("Đã tải cấu hình vgg_transformer")
    except Exception as e:
        logger.warning(f"Lỗi khi tải cấu hình vgg_transformer: {str(e)}. Thử resnet_transformer...")
        try:
            config = Cfg.load_config_from_name('resnet_transformer')  # Thử mô hình khác
            logger.info("Đã tải cấu hình resnet_transformer")
        except Exception as e2:
            logger.error(f"Lỗi khi tải cấu hình resnet_transformer: {str(e2)}")
            raise

    # Kiểm tra và khởi tạo config['data'] nếu thiếu
    if 'data' not in config:
        logger.warning("Khóa 'data' không tồn tại trong cấu hình. Khởi tạo mặc định...")
        config['data'] = {}
    
    config['device'] = 'cuda' if use_gpu else 'cpu'
    config['data']['imgH'] = 32
    config['data']['imgW'] = 200
    config['predictor']['beamsearch'] = True
    logger.info(f"Cấu hình: device={config['device']}, imgH={config['data']['imgH']}, imgW={config['data']['imgW']}")

    logger.info("Đang khởi tạo Predictor...")
    try:
        predictor = Predictor(config)
        logger.info("Đã khởi tạo Predictor thành công")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo Predictor: {str(e)}")
        raise

    return predictor

def recognize_text_vietocr(image_path, predictor):
    """
    Nhận dạng văn bản từ ảnh bằng VietOCR.
    """
    logger.info(f"Nhận dạng văn bản từ ảnh: {image_path}")
    img = Image.open(image_path).convert('RGB')
    text = predictor.predict(img)
    logger.info(f"Kết quả nhận dạng: {text}")
    return text