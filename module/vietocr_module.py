# module/vietocr_module.py
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image  # Thêm dòng này

def load_vietocr_model(use_gpu=False):
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cuda' if use_gpu else 'cpu'
    return Predictor(config)

def recognize_text_vietocr(image_path, predictor):
    img = Image.open(image_path).convert('RGB')
    return predictor.predict(img)