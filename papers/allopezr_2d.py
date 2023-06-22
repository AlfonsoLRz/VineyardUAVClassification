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
    #in_pixel = Input(shape=(img_size[2],))
    x = in_patch

    # y = Reshape([img_size[0] * img_size[1], img_size[2]])(x)
    # y = Lambda(lambda z: K.l2_normalize(z, axis=-1))(y)
    # y = SpatialAttention()(y)
    # y = Reshape([img_size[0], img_size[1], img_size[2]])(y)
    # x = Concatenate(axis=3)([x, y])

    x = Conv2D(start_size * 1, 1, strides=1, padding="same")(x)
    x = Conv2D(start_size * 1, kernel_size, strides=strides, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = get_inception_module(x, start_size * 2, strides=strides)
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


def get_kt_allopezr_2d_model(hp):
    in_patch = Input(shape=(patch_size, patch_size, num_target_features))
    x = in_patch

    conv_1a = hp.Int('conv_1a', min_value=16, max_value=96, step=16)
    conv_2b = hp.Int('conv_2b', min_value=16, max_value=128, step=16)
    conv_3c = hp.Int('conv_3c', min_value=16, max_value=128, step=16)

    use_first_conv = hp.Boolean('use_first_conv')
    lr = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
    intermediate_dropout = hp.Float('intermediate_dropout', min_value=0.1, max_value=0.5, step=0.1)
    first_kernel_size = hp.Choice('first_kernel_size', values=[3, 5, 7])
    final_dropout = hp.Float('final_dropout', min_value=0.1, max_value=0.5, step=0.1)
    leaky_alpha = hp.Float('leaky_alpha', min_value=0.1, max_value=0.3, step=0.1)
    activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid', 'softmax', 'elu', 'selu', 'softplus',
                                                 'softsign', 'hard_sigmoid', 'exponential'])
    strides_hp = hp.Choice('strides', values=[1, 2, 3])

    if use_first_conv:
        x = Conv2D(conv_1a, 1, strides=strides_hp, padding="same")(x)
    x = Conv2D(conv_1a, first_kernel_size, strides=strides_hp, padding="same")(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = get_naive_inception_module(x, conv_2b, strides=strides_hp)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = BatchNormalization()(x)
    x = Dropout(intermediate_dropout)(x)
    x = get_naive_inception_module(x, conv_3c, strides=strides_hp)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(final_dropout)(x)
    outputs = Dense(9, activation=activation)(x)
    model = Model([in_patch], outputs)

    model.compile(optimizer=RMSprop(lr=lr), loss=loss, metrics=keras.metrics.SparseCategoricalAccuracy())

    return model
