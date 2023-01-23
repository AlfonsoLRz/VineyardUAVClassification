import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import time



def get_dl_callback_list(exp_id, patch_size, start_size, monitor_early_stopping='sparse_categorical_accuracy', patience=3):
    return [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.EarlyStopping(
            monitor=monitor_early_stopping,
            patience=patience,
            verbose=3,
            mode='auto'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='Net/Best_' + exp_id + '.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        TimeCallback(),
    ]


def render_confusion_matrix(Y_test, Y_pred):
    flatten_y_test = np.reshape(Y_test, (-1,))
    flatten_y_pred = np.reshape(Y_pred, (-1,))

    num_classes = np.max(Y_test) + 1
    cm = confusion_matrix(flatten_y_test, flatten_y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # sn.set(font_scale=1)
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='g')
    # sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)

    sns.heatmap(cmn, annot=True, fmt='.2f', cmap='magma', )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('ConfusionMatrix.png')
    plt.show(block=False)
