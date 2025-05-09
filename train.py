import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tắt oneDNN
import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(8)  # Tăng thread CPU
tf.config.threading.set_inter_op_parallelism_threads(8)
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.mixed_precision import set_global_policy  # Tắt mixed_float16
from module.crnn_model import build_model
from preprocess import load_data
import logging
from collections import Counter
import json

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"TensorFlow version: {tf.__version__}")

# Hàm mất mát CTC
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Hàm làm sạch nhãn
def clean_label(label, valid_chars):
    return ''.join(c for c in label if c in valid_chars)

# Kiểm tra sự tồn tại của thư mục và file
def check_data_paths(data_dir, label_file):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Thư mục {data_dir} không tồn tại. Vui lòng tạo thư mục và đặt ảnh vào đó.")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"File {label_file} không tồn tại. Vui lòng tạo file với định dạng 'tên_ảnh:văn_bản'.")

# Tải dữ liệu
logger.info("Đang tải dữ liệu huấn luyện và kiểm tra...")
train_data_dir = 'D:/Vietnamese-handwritten/data/train'
train_label_file = 'D:/Vietnamese-handwritten/data/train/labels.txt'
test_data_dir = 'D:/Vietnamese-handwritten/data/test'
test_label_file = 'D:/Vietnamese-handwritten/data/test/labels.txt'

check_data_paths(train_data_dir, train_label_file)
check_data_paths(test_data_dir, test_label_file)

train_images, train_labels = load_data(train_data_dir, train_label_file)
test_images, test_labels = load_data(test_data_dir, test_label_file)

# Kiểm tra kích thước ảnh
expected_shape = (118, 2167, 1)
if train_images.shape[1:] != expected_shape:
    raise ValueError(f"Kích thước train_images không khớp: {train_images.shape[1:]} != {expected_shape}")
if test_images.shape[1:] != expected_shape:
    raise ValueError(f"Kích thước test_images không khớp: {test_images.shape[1:]} != {expected_shape}")

# Kiểm tra dữ liệu
logger.info(f"Số mẫu huấn luyện: {len(train_images)}")
logger.info(f"Số mẫu kiểm tra: {len(test_images)}")
logger.info(f"Mẫu nhãn huấn luyện (10 nhãn đầu tiên): {train_labels[:10]}")
all_chars = ''.join(train_labels + test_labels)
char_counts = Counter(all_chars)
logger.info(f"Phân phối ký tự trong nhãn: {dict(char_counts)}")

# Chuẩn hóa ảnh về [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
logger.info(f"train_images shape: {train_images.shape}")
logger.info(f"test_images shape: {test_images.shape}")

# Bộ từ vựng
characters = (
    '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'
    'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ'
    'Ơ'
    ' .,:!?()"-/[]\'%;@#$&*+={}|\<>~^_\t\n'
)
char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
logger.info(f"Số ký tự trong characters: {len(characters)}")
missing_chars = set(char_counts.keys()) - set(characters)
logger.info(f"Ký tự trong nhãn nhưng không có trong characters: {missing_chars}")
missing_in_data = set(characters) - set(char_counts.keys())
logger.info(f"Ký tự trong characters nhưng không có trong nhãn: {missing_in_data}")

# Lưu char_to_idx vào file JSON
with open('D:/Vietnamese-handwritten/data/char_to_idx.json', 'w', encoding='utf-8') as f:
    json.dump(char_to_idx, f, ensure_ascii=False)
logger.info("Đã lưu char_to_idx vào D:/Vietnamese-handwritten/data/char_to_idx.json")

# Làm sạch và chuẩn hóa nhãn
train_labels = [clean_label(label.lower(), characters) for label in train_labels]
test_labels = [clean_label(label.lower(), characters) for label in test_labels]

# Kiểm tra ký tự
for label in train_labels + test_labels:
    for char in label:
        if char not in char_to_idx:
            logger.error(f"Ký tự không có trong char_to_idx: '{char}' trong nhãn: '{label}'")
            raise ValueError(f"Ký tự '{char}' không có trong bộ từ vựng!")

# Mã hóa nhãn
max_label_len = max(len(label) for label in train_labels + test_labels)
logger.info(f"max_label_len: {max_label_len}")
train_labels_idx = np.array([[char_to_idx[c] for c in label] + [0] * (max_label_len - len(label)) for label in train_labels], dtype=np.int32)
test_labels_idx = np.array([[char_to_idx[c] for c in label] + [0] * (max_label_len - len(label)) for label in test_labels], dtype=np.int32)

# Tạo mô hình
logger.info("Tạo mô hình CRNN...")
model = build_model(num_classes=len(characters) + 1)
logger.info(f"Model output shape: {model.output_shape}")

# Tải mô hình từ checkpoint nếu tồn tại
if os.path.exists('D:/Vietnamese-handwritten/data/best_model.keras'):
    logger.info("Tải mô hình từ checkpoint...")
    model.load_weights('D:/Vietnamese-handwritten/data/best_model.keras')

# Chuẩn bị đầu vào
train_input_length = np.ones((len(train_images), 1), dtype=np.int32) * int(model.output_shape[1])
test_input_length = np.ones((len(test_images), 1), dtype=np.int32) * int(model.output_shape[1])
train_label_length = np.array([[len(label)] for label in train_labels], dtype=np.int32)
test_label_length = np.array([[len(label)] for label in test_labels], dtype=np.int32)

# Tăng cường dữ liệu
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

# Xác định đầu vào cho mô hình huấn luyện
labels = Input(name='the_labels', shape=[max_label_len], dtype='int32')
input_length = Input(name='input_length', shape=[1], dtype='int32')
label_length = Input(name='label_length', shape=[1], dtype='int32')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    [model.output, labels, input_length, label_length]
)

# Biên dịch mô hình
training_model = Model(inputs=[model.input, labels, input_length, label_length], outputs=loss_out)
training_model.compile(optimizer=Adam(learning_rate=1e-4), loss={'ctc': lambda y_true, y_pred: y_pred})

# Đảm bảo thư mục đầu ra
os.makedirs('D:/Vietnamese-handwritten/data', exist_ok=True)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint('D:/Vietnamese-handwritten/data/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# Batch generator
def batch_generator(images, labels_idx, input_length, label_length, batch_size, augment=True):
    num_samples = len(images)
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            batch_images = images[batch_indices].copy()
            if augment:
                for i in range(len(batch_images)):
                    batch_images[i] = datagen.random_transform(batch_images[i]).astype(np.float32)
            batch_labels = labels_idx[batch_indices]
            batch_input_length = input_length[batch_indices]
            batch_label_length = label_length[batch_indices]
            inputs = {
                'crnn_input': batch_images,
                'the_labels': batch_labels,
                'input_length': batch_input_length,
                'label_length': batch_label_length
            }
            outputs = np.zeros([len(batch_indices)], dtype=np.float32)
            logger.info(f"Xử lý batch {start//batch_size + 1}/{num_samples//batch_size + 1}")
            yield (inputs, outputs)

# Huấn luyện
batch_size = 8  # Giảm batch_size
train_generator = batch_generator(train_images, train_labels_idx, train_input_length, train_label_length, batch_size, augment=True)
validation_generator = batch_generator(test_images, test_labels_idx, test_input_length, test_label_length, batch_size, augment=False)
logger.info("Bắt đầu huấn luyện mô hình...")
training_model.fit(
    train_generator,
    steps_per_epoch=(len(train_images) + batch_size - 1) // batch_size,
    validation_data=validation_generator,
    validation_steps=(len(test_images) + batch_size - 1) // batch_size,
    epochs=20,
    callbacks=callbacks,
    verbose=1  # Hiển thị tiến trình
)

# Lưu mô hình
logger.info("Lưu mô hình cuối cùng...")
model.save('D:/Vietnamese-handwritten/data/final_model.keras')
logger.info("Đã lưu mô hình tại D:/Vietnamese-handwritten/data/final_model.keras")