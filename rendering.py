from config import *
import matplotlib.pyplot as plt
import numpy as np
from paths import *
from randomness import *
import umap.umap_ as umap


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


def render_hc_spectrum_label(hc_numpy, mask):
    """
    Renders the spectrum of every label in the hypercube.
    """
    n_classes = np.unique(mask)

    for class_id in n_classes:
        sample_subset = hc_numpy[mask == class_id, :]
        plt.plot(np.average(sample_subset, axis=0), label=class_id)
    plt.legend()
    plt.show()


def render_mask_histogram(label):
    plt.hist(label.flatten(), bins='auto', rwidth=1.0)
    plt.show()


def render_model_history(history, model_name, accuracy="sparse_categorical_accuracy"):
    # Accuracy
    h_epochs = range(1, len(history.history['loss']) + 1)

    plt.plot(h_epochs, history.history[accuracy], label="Training Accuracy")
    plt.plot(history.history['val_' + accuracy], label="Validation Accuracy")
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.figure()

    # Loss
    plt.plot(h_epochs, history.history['loss'], label="Training loss")
    plt.plot(h_epochs, history.history['val_loss'], label="Validation loss")
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(paths.result_folder + model_name + '.png')
    plt.show()


def render_umap_spectrum(patch, label):
    """
    Renders the UMAP of the spectrum of every label in the hypercube.
    """
    reducer = umap.UMAP(random_state=random_seed)
    embedding = reducer.fit_transform(patch)

    # Adjust size of plot
    plt.figure(figsize=(10, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.tight_layout()
    plt.show()
