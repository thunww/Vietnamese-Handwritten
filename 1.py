import tensorflow as tf

# Kiểm tra số lượng GPU có sẵn
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Kiểm tra chi tiết về các GPU có sẵn
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"GPU: {gpu}")
else:
    print("No GPU found")
