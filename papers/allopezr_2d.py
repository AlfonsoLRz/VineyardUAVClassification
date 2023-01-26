from keras.backend import int_shape
from keras.models import Model
from keras.layers import Concatenate, BatchNormalization, Input, Dense, Reshape, Activation, Dropout, Conv2D, \
    LeakyReLU, Flatten, GRU, MaxPooling2D


def get_residual_block(input, start_size, strides):
    conv1 = Conv2D(start_size, (3, 3), strides=strides, padding='same', activation='relu')(input)
    conv2 = Conv2D(start_size, (3, 3), strides=strides, padding='same', activation='relu')(conv1)
    output = Concatenate(axis=3)([conv2, input])
    return output


def get_inception_module(input, start_size, strides):
    conv1_a = Conv2D(start_size, (1, 1), strides=strides, padding='same', activation='relu')(input)
    conv2_a = Conv2D(start_size, (3, 3), strides=strides, padding='same', activation='relu')(conv1_a)

    conv1_b = Conv2D(start_size, (1, 1), strides=strides, padding='same', activation='relu')(input)
    conv2_b = Conv2D(start_size, (5, 5), strides=strides, padding='same', activation='relu')(conv1_b)

    max_pool_c = MaxPooling2D((3, 3), strides=strides, padding='same')(input)
    conv1_c = Conv2D(start_size, (1, 1), padding='same', strides=strides, activation='relu')(max_pool_c)

    output = Concatenate(axis=3)([conv2_a, conv2_b, conv1_c])
    return output


def get_allopezr_2d_model(img_size, num_classes, start_size=32, intermediate_activation='relu', kernel_size=3,
                          strides=2):
    in_patch = Input(shape=img_size)
    #in_pixel = Input(shape=(img_size[2],))

    residual = in_patch
    x = Conv2D(start_size * 1, 1, strides=strides, padding="same")(in_patch)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(start_size * 2, kernel_size, strides=strides, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    residual = Conv2D(start_size * 2, 1, strides=strides * 2, padding="same")(residual)
    x = Concatenate(axis=3)([x, residual])
    x = Conv2D(start_size * 4, kernel_size, strides=strides, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model([in_patch], outputs)

    return model
