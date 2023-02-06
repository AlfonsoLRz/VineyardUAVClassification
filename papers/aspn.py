from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, Dropout, Layer, BatchNormalization
from keras.initializers import Initializer
import math


class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[input_shape[-2], 1], name='kernel', initializer='ones', trainable=True)
        self.bias = self.add_weight(shape=[input_shape[-2]], name='bias', initializer='zeros', trainable=True)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        mid = input_shape[-2] // 2

        coe = K.l2_normalize(K.batch_dot(inputs, K.permute_dimensions(inputs, pattern=(0, 2, 1))), axis=-1)
        coe0 = K.expand_dims(coe[:, mid, :], axis=-1) * self.kernel
        w = K.batch_dot(coe, coe0) + K.expand_dims(self.bias, axis=-1)
        outputs = K.softmax(w, axis=-2) * inputs

        return outputs

    def get_config(self):
        config = {}
        base_config = super(SpatialAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)


class SecondOrderPooling(Layer):
    def __init__(self,
                 **kwargs):
        super(SecondOrderPooling, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)

        outputs = K.batch_dot(K.permute_dimensions(inputs, pattern=(0, 2, 1)), inputs, axes=[2, 1])
        outputs = K.reshape(outputs, [-1, input_shape[2] * input_shape[2]])

        return outputs

    def get_config(self):
        config = {}
        base_config = super(SecondOrderPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 4:
            output_shape = list([None, input_shape[1], input_shape[3] * input_shape[3]])
        else:
            output_shape = list([None, input_shape[2] * input_shape[2]])
        return tuple(output_shape)


def get_aspn_model(img_size, nb_classes, type='aspn'):
    if type == 'spn':
        model = spn(img_size, nb_classes)
    else:
        model = aspn(img_size, nb_classes)

    return model


class Symmetry(Initializer):
    """N*N*C Symmetry initial
    """

    def __init__(self, n=200, c=16, seed=0):
        self.n = n
        self.c = c
        self.seed = seed

    def __call__(self, shape, dtype=None):
        rv = K.truncated_normal([self.n, self.n, self.c], 0., 1e-5, dtype=dtype, seed=self.seed)
        rv = (rv + K.permute_dimensions(rv, pattern=(1, 0, 2))) / 2.0
        return K.reshape(rv, [self.n * self.n, self.c])


def spn(img_size, nb_classes):
    CNNInput = Input(shape=img_size, name='i0')

    F = Reshape([img_size[0] * img_size[1], img_size[2]])(CNNInput)
    F = BatchNormalization()(F)
    F = Dropout(rate=0.5)(F)

    F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='f2')(F)
    F = SecondOrderPooling(name='feature1')(F)
    F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='feature2')(F)

    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=img_size[2],
                                                                                               c=nb_classes))(F)
    model = Model(inputs=[CNNInput], outputs=F)

    return model


def aspn(img_size, nb_classes):
    CNNInput = Input(shape=img_size, name='i0')

    F = Reshape([img_size[0] * img_size[1], img_size[2]])(CNNInput)
    F = BatchNormalization()(F)
    F = Dropout(rate=0.5)(F)

    F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='f2')(F)
    F = SpatialAttention(name='f3')(F)
    F = SecondOrderPooling(name='feature1')(F)
    F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='feature2')(F)

    n = math.ceil(math.sqrt(K.int_shape(F)[-1]))
    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=n, c=nb_classes))(F)
    model = Model(inputs=[CNNInput], outputs=F)

    return model
