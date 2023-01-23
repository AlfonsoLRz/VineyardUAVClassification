from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Concatenate, Cropping2D  # CenterCrop
from tensorflow.keras.layers import Conv2D, Conv3D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense, Reshape
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.regularizers import l2, l1, L1L2

L2 = 0.001  # L2:0.002
act_reg = L1L2(l1=0.00001, l2=0.00001)  # L1L2(l1=0.0001, l2=0.0001) # default (0.01, 0.01)


def jigsaw_m2(input_net, internal_size=13):
    # Creates internal filters as Inception: 1x1, 3x3, 5x5 ..., nxn
    # Where n = internal_size

    jigsaw_t1_3x3_reduce = Conv2D(16, (1, 1), padding='same', activation='relu', kernel_regularizer=l2(L2),
                                  activity_regularizer=act_reg)(input_net)
    jigsaw_t1_3x3 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(L2),
                           activity_regularizer=act_reg)(jigsaw_t1_3x3_reduce)  # , name="i_3x3"

    if internal_size >= 5:
        jigsaw_t1_5x5_reduce = Conv2D(16, (1, 1), padding='same', activation='relu',
                                      kernel_regularizer=l2(L2),
                                      activity_regularizer=act_reg)(input_net)
        jigsaw_t1_5x5 = Conv2D(16, (5, 5), padding='same', activation='relu',
                               kernel_regularizer=l2(L2),
                               activity_regularizer=act_reg)(jigsaw_t1_5x5_reduce)  # , name="i_5x5"
    if internal_size >= 7:
        jigsaw_t1_7x7_reduce = Conv2D(16, (1, 1), padding='same', activation='relu',
                                      kernel_regularizer=l2(L2),
                                      activity_regularizer=act_reg)(input_net)
        jigsaw_t1_7x7 = Conv2D(16, (7, 7), padding='same', activation='relu',
                               kernel_regularizer=l2(L2),
                               activity_regularizer=act_reg)(jigsaw_t1_7x7_reduce)  # , name="i_7x7"
    if internal_size >= 9:
        jigsaw_t1_9x9_reduce = Conv2D(16, (1, 1), padding='same', activation='relu',
                                      kernel_regularizer=l2(L2),
                                      activity_regularizer=act_reg)(input_net)
        jigsaw_t1_9x9 = Conv2D(16, (9, 9), padding='same', activation='relu',
                               kernel_regularizer=l2(L2),
                               activity_regularizer=act_reg)(jigsaw_t1_9x9_reduce)  # , name="i_9x9"
    if internal_size >= 11:
        jigsaw_t1_11x11_reduce = Conv2D(16, (1, 1), padding='same', activation='relu',
                                        kernel_regularizer=l2(L2),
                                        activity_regularizer=act_reg)(input_net)
        jigsaw_t1_11x11 = Conv2D(16, (11, 11), padding='same', activation='relu',
                                 kernel_regularizer=l2(L2),
                                 activity_regularizer=act_reg)(jigsaw_t1_11x11_reduce)  # , name="i_11x11"
    if internal_size >= 13:
        jigsaw_t1_13x13_reduce = Conv2D(16, (1, 1), padding='same', activation='relu',
                                        kernel_regularizer=l2(L2),
                                        activity_regularizer=act_reg)(input_net)
        jigsaw_t1_13x13 = Conv2D(16, (13, 13), padding='same', activation='relu',
                                 kernel_regularizer=l2(L2),
                                 activity_regularizer=act_reg)(jigsaw_t1_13x13_reduce)  # , name="i_13x13"

    # Max. pool
    jigsaw_t1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_net)
    jigsaw_t1_pool_proj = Conv2D(32, (1, 1), padding='same', activation='relu',
                                 kernel_regularizer=l2(L2),
                                 activity_regularizer=act_reg)(jigsaw_t1_pool)

    # jigsaw_list = [jigsaw_t1_1x1, jigsaw_t1_3x3]
    jigsaw_list = [jigsaw_t1_3x3, jigsaw_t1_pool_proj]
    if internal_size >= 5:
        jigsaw_list.append(jigsaw_t1_5x5)
    if internal_size >= 7:
        jigsaw_list.append(jigsaw_t1_7x7)
    if internal_size >= 9:
        jigsaw_list.append(jigsaw_t1_9x9)
    if internal_size >= 11:
        jigsaw_list.append(jigsaw_t1_11x11)
    if internal_size >= 13:
        jigsaw_list.append(jigsaw_t1_13x13)

    # Add Conv1D
    jigsaw_t1_first = Conv2D(32, (1, 1), padding='same', activation='relu',
                             kernel_regularizer=l2(L2),
                             activity_regularizer=act_reg)(input_net)
    jigsaw_list.append(jigsaw_t1_first)

    if len(jigsaw_list) > 1:
        jigsaw_t1_output = Concatenate(axis=-1)(jigsaw_list)
    else:
        jigsaw_t1_output = jigsaw_list[0]

    avg_pooling = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), name='avg_pooling')(jigsaw_t1_output)
    flat = Flatten()(avg_pooling)
    flat = Dense(256, kernel_regularizer=l2(L2), activity_regularizer=act_reg)(flat)
    flat = Dropout(0.4)(flat)

    return flat


def jigsaw_m_end(module_b, first_layer=None, crop=True):
    if first_layer is not None and crop:
        # jigsaw_center  = CenterCrop(height = 1, width = 1)(first_layer)
        fl_shape = first_layer.shape
        if len(fl_shape) > 4:
            first_layer = Reshape((fl_shape[1], fl_shape[2], fl_shape[3] * fl_shape[4]))(first_layer)

        fl_shape = first_layer.shape
        crop_1 = fl_shape[1] // 2
        crop_2 = fl_shape[2] // 2
        jigsaw_center = Cropping2D(cropping=((crop_1, crop_1), (crop_2, crop_2)))(first_layer)
        input_pixel = Flatten()(jigsaw_center)
        input_pixel = Dense(16, kernel_regularizer=l2(L2),
                            activity_regularizer=act_reg)(input_pixel)
        input_pixel = Dropout(0.4)(input_pixel)
        input_pixel = Dense(16, kernel_regularizer=l2(L2),
                            activity_regularizer=act_reg)(input_pixel)
        input_pixel = Dropout(0.4)(input_pixel)
        flat = Concatenate(axis=-1)([input_pixel, module_b])
    else:
        flat = module_b

    flat = Dense(128, kernel_regularizer=l2(L2), activity_regularizer=act_reg)(flat)
    dropout = Dropout(0.4)(flat)

    return dropout


def build_jigsaw_hsi(num_classes, internal_size=13, image_dim=(19, 19, 7, 1), dimension_filters=None, crop=True):
    print(f"*** Building Jigsaw with up to {internal_size}x{internal_size} kernels")

    my_input = Input(shape=image_dim)

    # Normalize input data
    unit_norm = LayerNormalization(axis=2)(my_input)

    # Module A
    # conv_layer1 = Conv3D(filters=dimension_filters, kernel_size=(1, 1, 3), strides=(1, 1, 2), activation='relu', name='3d_1x1x7')(unit_norm)
    # conv_layer2 = Conv3D(filters=32, kernel_size=(1, 1, 5), strides=(1, 1, 2), activation='relu', name='3d_1x1x5')(conv_layer1)
    # conv_layer3 = Conv3D(filters=16, kernel_size=(1, 1, 3), strides=(1, 1, 2), activation='relu', name='3d_1x1x3')(unit_norm)
    conv_layer3 = unit_norm

    conv3d_shape = conv_layer3.shape
    if len(conv3d_shape) > 4:
        conv1 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3] * conv3d_shape[4]))(conv_layer3)
    else:
        conv1 = conv_layer3

    # Not needed for SA
    if (dimension_filters is None) or (dimension_filters < 1):
        # my_input_shape = my_input.shape
        # conv1 = Reshape((my_input_shape[1], my_input_shape[2], my_input_shape[3]*my_input_shape[4]))(my_input)
        conv1 = conv1
    else:
        conv1 = Conv2D(dimension_filters, (1, 1), padding='same', activation='relu', kernel_regularizer=l2(L2),
                       name='spectral_filter')(conv1)

    # Module B
    jigsaw_01 = jigsaw_m2(my_input if conv1 is None else conv1, internal_size=internal_size)
    # For SA, the next two lines must be uncommented
    # jigsaw_01 = jigsaw_m2( jigsaw_01, first_layer=my_input, internal_size = internal_size )
    # jigsaw_01 = jigsaw_m2( jigsaw_01, internal_size = internal_size )

    # Module C
    loss3_classifier_act = jigsaw_m_end(jigsaw_01, first_layer=my_input, crop=crop)  # testing num_classes
    output = Dense(num_classes, activation="softmax", name="dense_output")(loss3_classifier_act)

    model3 = Model(inputs=my_input, outputs=output, name='JigsawHSI')
    return model3
