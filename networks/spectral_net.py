from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from keras import backend as Kb
from keras.layers import Lambda
from keras.layers import Activation, AveragePooling2D, Concatenate


def wavelet_transform_axis_y(batch_img):
    odd_img = batch_img[:, 0::2]
    even_img = batch_img[:, 1::2]
    L = (odd_img + even_img) / 2.0
    H = Kb.abs(odd_img - even_img)
    return L, H


def wavelet_transform_axis_x(batch_img):
    # transpose + fliplr
    tmp_batch = Kb.permute_dimensions(batch_img, [0, 2, 1])[:, :, ::-1]
    _dst_L, _dst_H = wavelet_transform_axis_y(tmp_batch)
    # transpose + flipud
    dst_L = Kb.permute_dimensions(_dst_L, [0, 2, 1])[:, ::-1, ...]
    dst_H = Kb.permute_dimensions(_dst_H, [0, 2, 1])[:, ::-1, ...]
    return dst_L, dst_H


def Wavelet(batch_image):
    # make channel first image
    batch_image = Kb.permute_dimensions(batch_image, [0, 3, 1, 2])
    r = batch_image[:, 0]
    g = batch_image[:, 1]
    b = batch_image[:, 2]

    # level 1 decomposition
    wavelet_L, wavelet_H = wavelet_transform_axis_y(r)
    r_wavelet_LL, r_wavelet_LH = wavelet_transform_axis_x(wavelet_L)
    r_wavelet_HL, r_wavelet_HH = wavelet_transform_axis_x(wavelet_H)

    wavelet_L, wavelet_H = wavelet_transform_axis_y(g)
    g_wavelet_LL, g_wavelet_LH = wavelet_transform_axis_x(wavelet_L)
    g_wavelet_HL, g_wavelet_HH = wavelet_transform_axis_x(wavelet_H)

    wavelet_L, wavelet_H = wavelet_transform_axis_y(b)
    b_wavelet_LL, b_wavelet_LH = wavelet_transform_axis_x(wavelet_L)
    b_wavelet_HL, b_wavelet_HH = wavelet_transform_axis_x(wavelet_H)

    wavelet_data = [r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH,
                    g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH,
                    b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH]
    transform_batch = Kb.stack(wavelet_data, axis=1)

    # level 2 decomposition
    wavelet_L2, wavelet_H2 = wavelet_transform_axis_y(r_wavelet_LL)
    r_wavelet_LL2, r_wavelet_LH2 = wavelet_transform_axis_x(wavelet_L2)
    r_wavelet_HL2, r_wavelet_HH2 = wavelet_transform_axis_x(wavelet_H2)

    wavelet_L2, wavelet_H2 = wavelet_transform_axis_y(g_wavelet_LL)
    g_wavelet_LL2, g_wavelet_LH2 = wavelet_transform_axis_x(wavelet_L2)
    g_wavelet_HL2, g_wavelet_HH2 = wavelet_transform_axis_x(wavelet_H2)

    wavelet_L2, wavelet_H2 = wavelet_transform_axis_y(b_wavelet_LL)
    b_wavelet_LL2, b_wavelet_LH2 = wavelet_transform_axis_x(wavelet_L2)
    b_wavelet_HL2, b_wavelet_HH2 = wavelet_transform_axis_x(wavelet_H2)

    wavelet_data_l2 = [r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2,
                       g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2,
                       b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2]
    transform_batch_l2 = Kb.stack(wavelet_data_l2, axis=1)

    # level 3 decomposition
    wavelet_L3, wavelet_H3 = wavelet_transform_axis_y(r_wavelet_LL2)
    r_wavelet_LL3, r_wavelet_LH3 = wavelet_transform_axis_x(wavelet_L3)
    r_wavelet_HL3, r_wavelet_HH3 = wavelet_transform_axis_x(wavelet_H3)

    wavelet_L3, wavelet_H3 = wavelet_transform_axis_y(g_wavelet_LL2)
    g_wavelet_LL3, g_wavelet_LH3 = wavelet_transform_axis_x(wavelet_L3)
    g_wavelet_HL3, g_wavelet_HH3 = wavelet_transform_axis_x(wavelet_H3)

    wavelet_L3, wavelet_H3 = wavelet_transform_axis_y(b_wavelet_LL2)
    b_wavelet_LL3, b_wavelet_LH3 = wavelet_transform_axis_x(wavelet_L3)
    b_wavelet_HL3, b_wavelet_HH3 = wavelet_transform_axis_x(wavelet_H3)

    wavelet_data_l3 = [r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3,
                       g_wavelet_LL3, g_wavelet_LH3, g_wavelet_HL3, g_wavelet_HH3,
                       b_wavelet_LL3, b_wavelet_LH3, b_wavelet_HL3, b_wavelet_HH3]
    transform_batch_l3 = Kb.stack(wavelet_data_l3, axis=1)

    # level 4 decomposition
    wavelet_L4, wavelet_H4 = wavelet_transform_axis_y(r_wavelet_LL3)
    r_wavelet_LL4, r_wavelet_LH4 = wavelet_transform_axis_x(wavelet_L4)
    r_wavelet_HL4, r_wavelet_HH4 = wavelet_transform_axis_x(wavelet_H4)

    wavelet_L4, wavelet_H4 = wavelet_transform_axis_y(g_wavelet_LL3)
    g_wavelet_LL4, g_wavelet_LH4 = wavelet_transform_axis_x(wavelet_L4)
    g_wavelet_HL4, g_wavelet_HH4 = wavelet_transform_axis_x(wavelet_H4)

    wavelet_L3, wavelet_H3 = wavelet_transform_axis_y(b_wavelet_LL3)
    b_wavelet_LL4, b_wavelet_LH4 = wavelet_transform_axis_x(wavelet_L4)
    b_wavelet_HL4, b_wavelet_HH4 = wavelet_transform_axis_x(wavelet_H4)

    wavelet_data_l4 = [r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4,
                       g_wavelet_LL4, g_wavelet_LH4, g_wavelet_HL4, g_wavelet_HH4,
                       b_wavelet_LL4, b_wavelet_LH4, b_wavelet_HL4, b_wavelet_HH4]
    transform_batch_l4 = Kb.stack(wavelet_data_l4, axis=1)
    decom_level_1 = Kb.permute_dimensions(transform_batch, [0, 2, 3, 1])
    decom_level_2 = Kb.permute_dimensions(transform_batch_l2, [0, 2, 3, 1])
    decom_level_3 = Kb.permute_dimensions(transform_batch_l3, [0, 2, 3, 1])
    decom_level_4 = Kb.permute_dimensions(transform_batch_l4, [0, 2, 3, 1])

    return [decom_level_1, decom_level_2, decom_level_3, decom_level_4]


def wavelet_out_shape(input_shapes):
    # print('in to shape')
    return [tuple([None, 112, 112, 12]), tuple([None, 56, 56, 12]),
            tuple([None, 28, 28, 12]), tuple([None, 14, 14, 12])]


def get_wavelet_cnn_model(input_shape, num_classes, kernel_size, pool_size=7, start_size=64):
    input_ = Input(input_shape, name='the_input')
    # wavelet = Lambda(Wavelet, name='wavelet')
    wavelet = Lambda(Wavelet, wavelet_out_shape, name='wavelet')
    input_l1, input_l2, input_l3, input_l4 = wavelet(input_)

    # level one decomposition starts
    conv_1 = Conv2D(start_size, kernel_size=kernel_size, padding='same', name='conv_1')(input_l1)
    norm_1 = BatchNormalization(name='norm_1')(conv_1)
    relu_1 = Activation('relu', name='relu_1')(norm_1)

    conv_1_2 = Conv2D(start_size, kernel_size=kernel_size, strides=(2, 2), padding='same', name='conv_1_2')(relu_1)
    norm_1_2 = BatchNormalization(name='norm_1_2')(conv_1_2)
    relu_1_2 = Activation('relu', name='relu_1_2')(norm_1_2)

    # level two decomposition starts
    conv_a = Conv2D(filters=start_size, kernel_size=kernel_size, padding='same', name='conv_a')(input_l2)
    norm_a = BatchNormalization(name='norm_a')(conv_a)
    relu_a = Activation('relu', name='relu_a')(norm_a)

    # Concatenate level one and level two decomposition
    concatenate_level_2 = Concatenate(axis=-1)([relu_1_2, relu_a])
    conv_2 = Conv2D(start_size * 2, kernel_size=kernel_size, padding='same', name='conv_2')(concatenate_level_2)
    norm_2 = BatchNormalization(name='norm_2')(conv_2)
    relu_2 = Activation('relu', name='relu_2')(norm_2)

    conv_2_2 = Conv2D(start_size * 2, kernel_size=kernel_size, strides=(2, 2), padding='same', name='conv_2_2')(relu_2)
    norm_2_2 = BatchNormalization(name='norm_2_2')(conv_2_2)
    relu_2_2 = Activation('relu', name='relu_2_2')(norm_2_2)

    # level three decomposition starts
    conv_b = Conv2D(filters=start_size * 2, kernel_size=kernel_size, padding='same', name='conv_b')(input_l3)
    norm_b = BatchNormalization(name='norm_b')(conv_b)
    relu_b = Activation('relu', name='relu_b')(norm_b)

    conv_b_2 = Conv2D(start_size * 2, kernel_size=kernel_size, padding='same', name='conv_b_2')(relu_b)
    norm_b_2 = BatchNormalization(name='norm_b_2')(conv_b_2)
    relu_b_2 = Activation('relu', name='relu_b_2')(norm_b_2)

    # Concatenate level two and level three decomposition
    concatenate_level_3 = Concatenate(axis=-1)([relu_2_2, relu_b_2])
    conv_3 = Conv2D(start_size * 4, kernel_size=kernel_size, padding='same', name='conv_3')(concatenate_level_3)
    norm_3 = BatchNormalization(name='norm_3')(conv_3)
    relu_3 = Activation('relu', name='relu_3')(norm_3)

    conv_3_2 = Conv2D(start_size * 4, kernel_size=kernel_size, strides=(2, 2), padding='same', name='conv_3_2')(relu_3)
    norm_3_2 = BatchNormalization(name='norm_3_2')(conv_3_2)
    relu_3_2 = Activation('relu', name='relu_3_2')(norm_3_2)

    # Level four decomposition start
    conv_c = Conv2D(start_size * 4, kernel_size=kernel_size, padding='same', name='conv_c')(input_l4)
    norm_c = BatchNormalization(name='norm_c')(conv_c)
    relu_c = Activation('relu', name='relu_c')(norm_c)

    conv_c_2 = Conv2D(start_size * 4, kernel_size=kernel_size, padding='same', name='conv_c_2')(relu_c)
    norm_c_2 = BatchNormalization(name='norm_c_2')(conv_c_2)
    relu_c_2 = Activation('relu', name='relu_c_2')(norm_c_2)

    conv_c_3 = Conv2D(start_size * 4, kernel_size=kernel_size, padding='same', name='conv_c_3')(relu_c_2)
    norm_c_3 = BatchNormalization(name='norm_c_3')(conv_c_3)
    relu_c_3 = Activation('relu', name='relu_c_3')(norm_c_3)

    # Concatenate level three and level four decomposition
    concatenate_level_4 = Concatenate(axis=-1)([relu_3_2, relu_c_3])
    conv_4 = Conv2D(start_size * 8, kernel_size=kernel_size, padding='same', name='conv_4')(concatenate_level_4)
    norm_4 = BatchNormalization(name='norm_4')(conv_4)
    relu_4 = Activation('relu', name='relu_4')(norm_4)

    conv_4_2 = Conv2D(start_size * 8, kernel_size=kernel_size, strides=(2, 2), padding='same', name='conv_4_2')(relu_4)
    norm_4_2 = BatchNormalization(name='norm_4_2')(conv_4_2)
    relu_4_2 = Activation('relu', name='relu_4_2')(norm_4_2)

    # conv_5_1 = Conv2D(128, kernel_size=kernel_size, padding='same', name='conv_5_1')(relu_4_2)
    # norm_5_1 = BatchNormalization(name='norm_5_1')(conv_5_1)
    # relu_5_1 = Activation('relu', name='relu_5_1')(norm_5_1)

    pool_5_1 = AveragePooling2D(pool_size=pool_size, strides=1, padding='same', name='avg_pool_5_1')(relu_4_2)
    # flat_5_1 = Flatten(name='flat_5_1')(pool_5_1)

    # fc_5 = Dense(2048, name='fc_5')(flat_5_1)
    # norm_5 = BatchNormalization(name='norm_5')(fc_5)
    # relu_5 = Activation('relu', name='relu_5')(norm_5)
    # drop_5 = Dropout(0.5, name='drop_5')(relu_5)

    # fc_6 = Dense(2048, name='fc_6')(drop_5)
    # norm_6 = BatchNormalization(name='norm_6')(fc_6)
    # relu_6 = Activation('relu', name='relu_6')(norm_6)
    # drop_6 = Dropout(0.5, name='drop_6')(relu_6)
    flatten_layer = Flatten()(pool_5_1)

    dense_layer1 = Dense(units=start_size * 4, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=start_size * 4, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=num_classes, activation='softmax')(dense_layer2)

    model = Model(inputs=input_, outputs=output_layer)

    return model
