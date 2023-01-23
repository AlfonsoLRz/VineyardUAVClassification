from keras.callbacks import LearningRateScheduler, Callback
import time
from tk.keras.optimizers import Adadelta, SGD, Adam, RMSprop


# Global config
loss = 'sparse_categorical_crossentropy'
metrics = ['sparse_categorical_accuracy']
batch_size = 32
epochs = 100
patch_size = 11
patch_overlapping = 9


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


training_config = {
    'aspn': {
        'optimizer': Adam(learning_rate=1e-1),
        'callbacks': [],
    },
    'fsk_net': {
        'optimizer': RMSprop(learning_rate=3e-4),
        'callbacks': [],
    },
    'hybrid_sn': {
        'optimizer': Adam(learning_rate=1e-3),
        'callbacks': [],
    },
    'jigsaw_hsi': {
        'optimizer': Adadelta(learning_rate=1e-2),
        'callbacks': [],
    },
    'lt_cnn': {
        'optimizer': Adam(learning_rate=1e-3, decay=1e-5),
        'callbacks': [LearningRateScheduler(decay_schedule, verbose=1)],
    },
    'spectral_net': {
        'optimizer': SGD(learning_rate=1e-2, momentum=0.9),
        'callbacks': [],
    },
    'ours_3d': {
        'optimizer': Adam(learning_rate=1e-3),
        'callbacks': [],
    }
}