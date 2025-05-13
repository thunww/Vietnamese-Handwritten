# module/vietnamese_ocr.py
import numpy as np
import cv2
from tensorflow.keras import backend as K

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def decode_predictions(preds, char_to_idx):
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    decoded = []
    for pred in preds:
        pred = np.argmax(pred, axis=-1)
        text = ''
        prev = None
        for idx in pred:
            if idx != prev and idx != 0:
                text += idx_to_char.get(idx, '')
            prev = idx
        decoded.append(text)
    return decoded

def recognize_text(image_path, model, char_to_idx):
    img = preprocess_image(image_path)
    preds = model.predict(img)
    text = decode_predictions(preds, char_to_idx)
    return text[0]