# ocr_app.py
import logging
import json
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.backend import ctc_decode
from module.crnn_model import build_model
from module.vietocr_module import load_vietocr_model, recognize_text_vietocr
from preprocess import preprocess_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"TensorFlow version: {tf.__version__}")

def load_custom_model(custom_model_path, num_classes):
    """
    Tải mô hình tùy chỉnh với kiến trúc được định nghĩa chính xác như khi đào tạo.
    
    Args:
        custom_model_path: Đường dẫn đến file mô hình đã lưu
        num_classes: Số lượng lớp đầu ra
        
    Returns:
        model: Mô hình đã tải trọng số
    """
    logger.info(f"Đang tải mô hình từ {custom_model_path}...")
    if not os.path.exists(custom_model_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình tại {custom_model_path}")
    
    try:
        # Thay vì sử dụng build_model từ module.crnn_model, định nghĩa lại kiến trúc
        # để đảm bảo nó khớp chính xác với mô hình đã đào tạo
        inputs = tf.keras.Input(shape=(118, 2167, 1))
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Reshape(target_shape=(-1, x.shape[-1]))(x)
        
        # Đảm bảo thông số LSTM chính xác
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True, 
                                 kernel_initializer='he_normal', 
                                 recurrent_initializer='orthogonal')
        )(x)
        
        x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs=x)
        
        logger.info("Đã xây dựng mô hình với kiến trúc tương thích")
        
        # Tải trọng số
        model.load_weights(custom_model_path)
        logger.info("Đã tải trọng lượng mô hình thành công")
        
        return model
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình: {str(e)}")
        # Thử phương pháp khác nếu phương pháp trên thất bại
        try:
            # Khởi tạo từng lớp với đúng kích thước
            logger.info("Thử phương pháp khác để tải mô hình...")
            # Reset default graph để tránh xung đột
            tf.compat.v1.reset_default_graph()
            
            # Tạo mô hình mới với cùng kích thước trọng số
            inputs = tf.keras.Input(shape=(118, 2167, 1))
            x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            
            # Lấy kích thước sau pooling
            _, height, width, channels = x.shape
            x = tf.keras.layers.Reshape(target_shape=(height * width, channels))(x)
            
            # Sử dụng LSTM với kích thước đúng
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(256, return_sequences=True)
            )(x)
            
            x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            model = tf.keras.Model(inputs, outputs=x)
            
            # Thử tải bằng load_model thay vì load_weights
            if custom_model_path.endswith('.keras'):
                model_dir = os.path.dirname(custom_model_path)
                model_name = os.path.basename(custom_model_path).replace('.keras', '')
                model_path = os.path.join(model_dir, model_name)
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path, compile=False)
                else:
                    # Thử tải trọng số một lần nữa
                    model.load_weights(custom_model_path)
            else:
                model.load_weights(custom_model_path)
                
            logger.info("Đã tải mô hình thành công bằng phương pháp thay thế")
            return model
        except Exception as e2:
            logger.error(f"Cả hai phương pháp đều thất bại: {str(e2)}")
            raise e

class OCR:
    def __init__(self, char_to_idx_path, custom_model_path=None):
        with open(char_to_idx_path, 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {int(idx): char for char, idx in self.char_to_idx.items()}
        self.vietocr_predictor = None
        self.custom_model = None
        
        try:
            logger.info("Đang tải mô hình VietOCR...")
            self.vietocr_predictor = load_vietocr_model(use_gpu=False)
            logger.info("Đã tải xong mô hình VietOCR")
        except Exception as e:
            logger.warning(f"Không thể tải mô hình VietOCR: {str(e)}")
        
        if custom_model_path:
            try:
                logger.info(f"Đang tải mô hình tự huấn luyện từ {custom_model_path}...")
                self.custom_model = load_custom_model(custom_model_path, len(self.char_to_idx) + 1)
                logger.info("Đã tải mô hình tự huấn luyện")
            except Exception as e:
                logger.warning(f"Không thể tải mô hình tự huấn luyện: {str(e)}")

    def preprocess_for_crnn(self, image_path, target_size=(118, 2167)):
        img = preprocess_image(image_path, target_size)
        if img is None:
            raise ValueError(f"Không thể xử lý ảnh từ {image_path}")
        return np.expand_dims(img, axis=0)

    def recognize(self, image_path, use_vietocr=True, preprocess=True):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh tại {image_path}")
        
        processed_image_path = image_path
        if preprocess:
            try:
                logger.info("Đang tiền xử lý ảnh...")
                processed_image = preprocess_image(image_path)
                filename = os.path.basename(image_path)
                dirname = os.path.dirname(image_path) or '.'
                processed_image_path = os.path.join(dirname, "processed_" + filename)
                processed_image.save(processed_image_path)
                logger.info(f"Đã lưu ảnh đã xử lý tại {processed_image_path}")
            except Exception as e:
                logger.error(f"Lỗi khi xử lý ảnh: {str(e)}")
                logger.warning("Sử dụng ảnh gốc...")
                processed_image = Image.open(image_path)
                processed_image_path = image_path
        
        if use_vietocr:
            if self.vietocr_predictor is None:
                raise ValueError("VietOCR model không được tải.")
            logger.info("Sử dụng VietOCR để nhận dạng...")
            return recognize_text_vietocr(processed_image_path, self.vietocr_predictor)
        else:
            if self.custom_model is None or self.idx_to_char is None:
                raise ValueError("Không có mô hình tự huấn luyện hoặc từ điển ký tự.")
            logger.info("Sử dụng mô hình tự train để nhận dạng...")
            
            img = self.preprocess_for_crnn(processed_image_path)
            y_pred = self.custom_model.predict(img)
            input_length = np.ones((1,)) * y_pred.shape[1]
            decoded, _ = ctc_decode(y_pred, input_length=input_length, greedy=True)
            
            prediction = []
            for seq in decoded[0].numpy():
                pred = ''.join(self.idx_to_char.get(idx, '') for idx in seq if idx != -1 and idx in self.idx_to_char)
                prediction.append(pred)
            
            return prediction[0]
        
def main():
    char_to_idx_path = "D:/Vietnamese-handwritten/data/char_to_idx_simplified.json"
    custom_model_path = "D:/Vietnamese-handwritten/data/new_model.keras"
    ocr = OCR(char_to_idx_path, custom_model_path=custom_model_path)
    image_path = input("Nhập đường dẫn đến file ảnh: ")
    use_vietocr = input("Sử dụng VietOCR? (y/n): ").lower() == 'y'
    try:
        text = ocr.recognize(image_path, use_vietocr)
        print(f"Văn bản được nhận dạng: {text}")
    except Exception as e:
        print(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main()