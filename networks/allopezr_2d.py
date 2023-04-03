from papers.aspn import SpatialAttention, SecondOrderPooling, Lambda
from config import loss, patch_size, num_target_features
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Concatenate, BatchNormalization, Input, Dense, Reshape, Activation, Dropout, Conv2D, \
    LeakyReLU, Flatten, MaxPooling2D, GaussianNoise, Lambda
from keras.optimizers import RMSprop


def get_residual_block(input, start_size, strides):
    conv1 = Conv2D(start_size, (3, 3), strides=strides, padding='same', activation='relu')(input)
    conv2 = Conv2D(start_size, (3, 3), strides=strides, padding='same', activation='relu')(conv1)
    output = Concatenate(axis=3)([conv2, input])
    return output


def get_inception_module(input, start_size, strides):
    conv1_a = Conv2D(start_size, (1, 1), strides=1, padding='same', activation='relu')(input)
    conv2_a = Conv2D(start_size, (3, 3), strides=strides, padding='same', activation='relu')(conv1_a)

    conv1_b = Conv2D(start_size, (1, 1), strides=1, padding='same', activation='relu')(input)
    conv2_b = Conv2D(start_size, (5, 5), strides=strides, padding='same', activation='relu')(conv1_b)

    max_pool_c = MaxPooling2D(strides, padding='same')(input)
    conv1_c = Conv2D(start_size, (1, 1), padding='same', strides=1, activation='relu')(max_pool_c)

    output = Concatenate(axis=3)([conv2_a, conv2_b, conv1_c])
    return output


def get_naive_inception_module(input, num_filters, strides=3):
    conv1x1 = Conv2D(num_filters, (1, 1), strides=strides, padding='same')(input)
    conv3x3 = Conv2D(num_filters, (3, 3), strides=strides, padding='same')(input)
    conv5x5 = Conv2D(num_filters, (5, 5), strides=strides, padding='same')(input)
    max_pool = MaxPooling2D((3, 3), strides=strides, padding='same')(input)

    return Concatenate(axis=3)([conv1x1, conv3x3, conv5x5, max_pool])


def get_allopezr_2d_model(img_size, num_classes, start_size=32, intermediate_activation='relu', kernel_size=3,
                          strides=2):
    in_patch = Input(shape=img_size)
    x = in_patch

    y = Reshape([img_size[0] * img_size[1], img_size[2]])(x)
    y = Lambda(lambda z: K.l2_normalize(z, axis=-1))(y)
    y = SpatialAttention()(y)
    y = Reshape([img_size[0], img_size[1], img_size[2]])(y)
    x = Concatenate(axis=3)([x, y])

    x = Conv2D(start_size * 1, 1, strides=1, padding="same")(x)
    x = Conv2D(start_size * 1, kernel_size, strides=strides, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = get_naive_inception_module(x, start_size * 2, strides=strides)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    x = get_inception_module(x, start_size * 6, strides=strides)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model([in_patch], outputs)

    return model