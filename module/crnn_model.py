import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def build_model(num_classes):
    inputs = Input(name='crnn_input', shape=(118, 2167, 1))
    
    # Block 1
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = MaxPool2D(pool_size=3, strides=3)(x)
    x = Activation('relu')(x)
    x_1 = x
    
    # Block 2
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = MaxPool2D(pool_size=3, strides=3)(x)
    x = Activation('relu')(x)
    x_2 = x
    
    # Block 3
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_3 = x
    
    # Block 4
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_3])
    x = Activation('relu')(x)
    
    # Block 5
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_5 = x
    
    # Block 6
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_5])
    x = Activation('relu')(x)
    
    # Block 7
    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 1))(x)
    x = Activation('relu')(x)
    
    # Pooling layer
    x = MaxPool2D(pool_size=(3, 1))(x)
    
    # Lambda layer to squeeze dimension
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)
    
    # Bidirectional LSTM layers
    blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(blstm_1)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax', name='dense')(blstm_2)
    
    # Define model
    model = Model(inputs, outputs)
    return model