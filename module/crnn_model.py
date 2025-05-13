import tensorflow as tf

def build_model(num_classes):
    # Explicitly name the input layer
    inputs = tf.keras.Input(shape=(64, 512, 1), name='input_layer')  # Added name='input_layer'
    
    # First Conv2D and Pooling
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second Conv2D and Pooling
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Reshape for LSTM
    x = tf.keras.layers.Reshape(target_shape=(-1, x.shape[-1]))(x)
    
    # Bidirectional LSTM layer
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
    
    # Dense layer for classification
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Define the model, specifying the output explicitly
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    return model