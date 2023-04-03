from tensorflow import keras
from tensorflow.keras import layers, regularizers


def get_nezami_model(img_size, num_classes, kernel_size=5, strides=3, start_size=20, intermediate_activation='relu'):
    inputs = keras.Input(shape=img_size)

    x = layers.Conv2D(start_size, kernel_size=kernel_size, strides=1)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=strides)(x)
    x = layers.Conv2D(50, kernel_size=kernel_size, strides=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=strides)(x)
    x = layers.Activation(intermediate_activation)(x)
    x = layers.Conv2D(3, kernel_size=1, strides=1)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    return model