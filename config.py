from keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras.optimizers import Adadelta, SGD, Adam, Adamax, RMSprop
import time


# Global config
batch_size = 256
epochs = 5
last_activation = 'softmax'
kernel_size = 3
loss = 'sparse_categorical_crossentropy'
patch_size = 23
patch_overlapping = 22
strides = 2
test_split = 0.2
validation_split = 0.1
num_target_features = 30


# Callbacks
def decay_schedule(epoch, lr):
    """
    Learning rate decay as iterations advance.
    """
    if (epoch % 5 == 0) and (epoch != 0):
        lr = lr * .9
    return lr


class TimeCallback(Callback):
    def __init__(self, logs={}):
        super().__init__()
        self.duration = 0
        self.timestamp = None

    def on_train_begin(self, logs=None):
        self.timestamp = time.time()

    def on_train_end(self, logs=None):
        self.duration = time.time() - self.timestamp
        print("Training took {} seconds".format(self.duration))


training_config = {
    'allopezr_2d': {
        'optimizer': Adam(learning_rate=1e-3),
        'callbacks': [],
        'intermediate_activation': 'relu',
        'kernel_size': 3,
        'start_size': 64,
        'strides': 2,
    },
    'allopezr_3d': {
        'optimizer': RMSprop(learning_rate=0.001),
        'callbacks': [],
        'intermediate_activation': 'relu',
        'kernel_size': 3,
        'start_size': 64,
        'strides': 2,
    },
    'aspn': {
        'optimizer': Adam(learning_rate=1e-1),
        'callbacks': [],
        'intermediate_activation': 'relu',
        'kernel_size': 3,
        'start_size': 32,
    },
    'fsk_net': {
        'optimizer': RMSprop(learning_rate=3e-4),
        'callbacks': [],
        'start_size': 16,
    },
    'hybrid_sn': {
        'optimizer': Adam(learning_rate=1e-3),
        'callbacks': [],
        'start_size': 8,
    },
    'jigsaw_hsi': {
        'optimizer': Adadelta(learning_rate=1e-2),
        'callbacks': [],
        'start_size': None,
        'kernel_size': 13,
    },
    'lt_cnn': {
        'optimizer': Adam(learning_rate=1e-3),
        'callbacks': [LearningRateScheduler(decay_schedule, verbose=1)],
        'start_size': 32,
        'kernel_size': 3,
        'strides': 2,
    },
    'nezami': {
        'optimizer': Adam(learning_rate=1e-3),
        'callbacks': [],
        'intermediate_activation': 'relu',
        'kernel_size': 3,
        'start_size': 16,
        'stride': 2,
    },
    'spectral_net': {
        'optimizer': SGD(learning_rate=1e-2, momentum=0.9),
        'callbacks': [],
        'start_size': 64,
        'kernel_size': 3,
        'strides': 7,
    },
    'ours_3d': {
        'optimizer': Adam(learning_rate=1e-3),
        'callbacks': [],
    }
}