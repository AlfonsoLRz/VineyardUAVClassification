from keras import backend as K
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, GlobalAveragePooling3D, AveragePooling3D, \
    BatchNormalization, Activation, Concatenate
from keras.models import Model


def dense_block(x, blocks, name, growth_rate=32):
    """
    A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
    return x


def conv_block(x, growth_rate, name):
    """
    A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv3D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv3D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def transition_block(x, reduction, name):
    """
    A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv3D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False, name=name + '_conv')(x)
    x = AveragePooling3D(1, strides=(2, 2, 2), name=name + '_pool')(x)
    return x


def get_fsk_model(img_size, num_classes, start_size=16):
    if len(img_size) != 4:
        raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

    input = Input(shape=img_size)
    conv1 = Conv3D(start_size, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                   kernel_initializer='he_normal')(input)
    pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(conv1)

    # Dense Block1
    x = dense_block(pool1, 2, growth_rate=start_size, name='conv1')
    x = transition_block(x, 0.5, name='pool1')
    # x = dense_block(x, 2, growth_rate=start_size, name='conv2')
    # x = transition_block(x, 0.5, name='pool2')
    # x = dense_block(x, 2, growth_rate=start_size, name='conv3')
    x = GlobalAveragePooling3D(name='avg_pool')(x)

    dense = Dense(units=num_classes, activation="softmax", kernel_initializer="he_normal")(x)
    model = Model(inputs=input, outputs=dense)

    return model
