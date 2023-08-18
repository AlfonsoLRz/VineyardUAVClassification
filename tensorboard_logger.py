import tensorflow as tf

class TensorboardLogger(tf.keras.callbacks.TensorBoard):
    def __init__(self, epoch_pad, log_dir, update_freq=10, histogram_freq=0, **kwargs):
        super().__init__(log_dir=log_dir, update_freq=update_freq, histogram_freq=histogram_freq, **kwargs)
        self.epoch_pad = epoch_pad

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(self.epoch_pad + epoch, logs=logs)