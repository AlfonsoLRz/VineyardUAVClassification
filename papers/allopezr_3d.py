from keras.models import Model
from keras.layers import Concatenate, BatchNormalization, Input, Dense, Reshape, Activation, Dropout, Conv2D, \
    LeakyReLU, Flatten, GRU, MaxPooling2D

def get_allopezr_3d_model(img_size, num_classes, start_size=32, intermediate_activation='relu', kernel_size=3,
                          strides=2):
    in_patch = Input(shape=img_size)
    in_pixel = Input(shape=(img_size[2],))

    x = Dense(img_size[0] * img_size[1], activation=intermediate_activation)(in_pixel)
    x = Reshape((img_size[0], img_size[1], 1))(x)
    merge = Concatenate()([in_patch, x])

    # x = Conv2D(start_size * 1, 1, strides=strides, padding="same")(merge)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = Conv2D(start_size * 2, kernel_size, strides=strides, padding="same")(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.4)(x)
    # x = Conv2D(start_size * 4, kernel_size, strides=strides, padding="same")(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = BatchNormalization()(x)
    # x = Flatten()(x)
    # x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model([in_patch, in_pixel], outputs)

    return model