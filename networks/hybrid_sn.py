from tensorflow import keras
from tensorflow.keras import layers, regularizers


def get_hybrid_sn_model(img_size, num_classes, start_size=8):
    inputs = keras.Input(shape=img_size)

    conv_layer1 = layers.Conv3D(filters=start_size, kernel_size=(3, 3, 7))(inputs)
    conv_layer2 = layers.Conv3D(filters=start_size * 2, kernel_size=(3, 3, 5))(conv_layer1)
    conv_layer3 = layers.Conv3D(filters=start_size * 4, kernel_size=(3, 3, 3))(conv_layer2)

    conv3d_shape = conv_layer3.shape
    conv_layer3 = layers.Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3] * conv3d_shape[4]))(conv_layer3)
    conv_layer4 = layers.Conv2D(filters=start_size * 8, kernel_size=(3, 3))(conv_layer3)

    flatten_layer = layers.Flatten()(conv_layer4)

    dense_layer1 = layers.Dense(units=start_size * 32)(flatten_layer)
    dense_layer1 = layers.Dropout(0.4)(dense_layer1)
    dense_layer2 = layers.Dense(units=start_size * 16)(dense_layer1)
    dense_layer2 = layers.Dropout(0.4)(dense_layer2)
    output_layer = layers.Dense(units=num_classes, activation='softmax')(dense_layer2)
    model = keras.Model(inputs, output_layer)

    return model