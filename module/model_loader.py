import cv2
import tensorflow as tf
import json
import os
import numpy as np 

def load_data(data_dir, label_file, image_size=(64, 512)):
    images = []
    labels = []
    
    # Đọc nhãn từ file labels.txt
    label_dict = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                label_dict[parts[0]] = parts[1]
    
    print("Image size:", image_size)
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                img = cv2.resize(img, (image_size[1], image_size[0]))
                img = img.reshape((image_size[0], image_size[1], 1))
                label = label_dict.get(filename, filename.split('.')[0])
                images.append(img)
                labels.append(label)
    
    images = np.array(images)
    return images, labels