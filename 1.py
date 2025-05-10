from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Đường dẫn đến file weights
model_path = 'D:/Vietnamese-handwritten/data/best_model.keras'

# Kiểm tra file có tồn tại không
if os.path.exists(model_path):
    print("✅ File tồn tại, bắt đầu load model...")

    # Xây dựng lại kiến trúc model giống khi training
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Load weights
    try:
        model.load_weights(model_path)
        print("✅ Load weights thành công.")
    except Exception as e:
        print("❌ Lỗi khi load weights:", e)
else:
    print("❌ File không tồn tại. Em nên train lại model từ đầu.")
