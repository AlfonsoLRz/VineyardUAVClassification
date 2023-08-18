from keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Concatenate
from keras.models import Model


def inception_module(x, num_filters_1, num_filters_2, num_filters_3, num_filters_4):
    # Branch 0
    max_pool_3x3_0 = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    conv_1x1_0 = Conv2D(num_filters_3, kernel_size=(1, 1))(max_pool_3x3_0)     # 96

    # Branch 1
    conv_1x1_1 = Conv2D(num_filters_3, kernel_size=(1, 1))(x)
    conv_3x3_1 = Conv2D(num_filters_4, kernel_size=2, padding='same')(conv_1x1_1)
    conv_3x3_1_2 = Conv2D(num_filters_4, kernel_size=2)(conv_3x3_1)

    # Branch 2
    conv_1x1_2 = Conv2D(num_filters_1, kernel_size=(1, 1))(x)
    conv_3x3_2 = Conv2D(num_filters_2, kernel_size=2)(conv_1x1_2)

    return Concatenate(axis=3)([conv_1x1_0, conv_3x3_1_2, conv_3x3_2])


def get_lt_cnn_model(img_size, num_classes, start_size=64, kernel_size=5, strides=2):
    input = Input(shape=img_size)

    conv_1a = Conv2D(start_size * 2, kernel_size=(5, 5), padding='same')(input)
    max_pool_0a = MaxPooling2D(pool_size=(2, 2), strides=1)(conv_1a)

    conv_2a = Conv2D(start_size * 3, kernel_size=(3, 3))(max_pool_0a)
    max_pool_1a = MaxPooling2D(pool_size=(2, 2), strides=1)(conv_2a)

    inception_3a = inception_module(max_pool_1a, start_size, start_size * 3, start_size * 3, start_size * 3)
    max_pool_2a = MaxPooling2D(pool_size=(2, 2), strides=1)(inception_3a)
    inception_3b = inception_module(max_pool_2a, start_size, start_size * 2, start_size * 3, start_size * 4)
    max_pool_3a = MaxPooling2D(pool_size=(3, 3), strides=2)(inception_3b)

    # Branch 0
    max_pool_3x3_0 = MaxPooling2D(pool_size=(2, 2), strides=1)(max_pool_3a)
    conv_1x1_0 = Conv2D(start_size * 7, kernel_size=(1, 1))(max_pool_3x3_0)

    # Branch 1
    conv_1x1_2 = Conv2D(start_size * 7, kernel_size=(1, 1))(max_pool_3a)
    conv_3x3_2 = Conv2D(start_size * 8, kernel_size=2)(conv_1x1_2)

    concatenate = Concatenate(axis=3)([conv_1x1_0, conv_3x3_2])

    max_pool_4 = MaxPooling2D(pool_size=(3, 3), strides=2)(concatenate)
    gap = GlobalAveragePooling2D()(max_pool_4)
    fully_connected = Dense(196)(gap)
    dropout = Dropout(0.4)(fully_connected)
    softmax = Dense(num_classes, activation='softmax')(dropout)
    model = Model(inputs=input, outputs=softmax)

    return model



