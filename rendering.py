import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker
import numpy as np
import paths
from randomness import *
import seaborn as sns
from sklearn.metrics import confusion_matrix
import training_history
import umap.umap_ as umap


def get_plot_fonts():
    font = 'Adobe Devanagari'
    title_font = {'fontname': font, 'size': 13}
    regular_font = {'fontname': font}
    font = font_manager.FontProperties(family=font, size=11)

    return font, title_font, regular_font


def render_confusion_matrix(y_test, y_pred):
    """
    Renders the confusion matrix of the model predictions.
    """
    flatten_y_test = np.reshape(y_test, (-1,))
    flatten_y_pred = np.reshape(y_pred, (-1,))

    cm = confusion_matrix(flatten_y_test, flatten_y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # sn.set(font_scale=1)
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='g')
    # sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)

    sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(paths.result_folder + 'ConfusionMatrix.png')
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


def render_label_diff(label_diff, filename, dpi=500):
    """
    Renders the difference between the ground truth and the predicted labels.
    """
    plt.imshow(label_diff, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(paths.result_folder + filename, dpi=dpi)
    plt.show()


def render_label_distribution(patches, labels):
    font, title_font, regular_font = get_plot_fonts()

    ground_indices = np.where(labels == 0)
    ground_patches = patches[ground_indices]
    vegetation_patches = patches[np.where(labels > 0)]

    sns.jointplot(x=ground_patches[:, 0], y=ground_patches[:, 1], kind='hex')
    plt.suptitle("Ground distribution")

    sns.jointplot(x=vegetation_patches[:, 0], y=vegetation_patches[:, 1], kind='hex')
    _ = plt.suptitle("Vegetation distribution")

    plt.tight_layout()
    plt.show()
    plt.savefig(paths.result_folder + 'label_distribution.png')


def render_mask_histogram(label):
    """
    Renders the histogram of the mask.
    """
    plt.hist(label.flatten(), bins='auto', rwidth=1.0)
    plt.show()


def render_model_history(history, model_name):
    """
    Renders the history of the model after training.
    """
    h_epochs = range(1, history.get_history_length() + 1)
    accuracy = history.get_accuracy_key()
    history_vector = history.get_history()

    plt.plot(h_epochs, history_vector[accuracy], label="Training Accuracy")
    plt.plot(h_epochs, history_vector['val_' + accuracy], label="Validation Accuracy")
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(paths.result_folder + model_name + '_accuracy.png')
    plt.figure()

    # Loss
    plt.plot(h_epochs, history_vector['loss'], label="Training loss")
    plt.plot(h_epochs, history_vector['val_loss'], label="Validation loss")
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(paths.result_folder + model_name + '_loss.png')
    plt.show()


## ---------------------------------------------
## -----------------  PATCHES  -----------------
## ---------------------------------------------

def get_batch_label(patch_labels, label):
    random_idx = np.random.randint(0, len(patch_labels))
    while patch_labels[random_idx] != label:
        random_idx = np.random.randint(0, len(patch_labels))

    return random_idx


def get_rgb_chunk(chunk):
    chunk_shape = chunk.shape
    return chunk[:, :, chunk_shape[2] - 1] - chunk[:, :, 0]


def get_rgb_mask(mask, max_labels):
    rgb_mask = np.dstack([mask, mask, mask])
    cmap = plt.cm.get_cmap('Spectral')

    for x in range(0, rgb_mask.shape[0]):
        for y in range(0, rgb_mask.shape[1]):
            color = cmap(rgb_mask[x, y, 0] / max_labels)
            rgb_mask[x, y, 0] = color[0]
            rgb_mask[x, y, 1] = color[1]
            rgb_mask[x, y, 2] = color[2]

    return rgb_mask


def plot_patch_variance(patch, patch_labels, axis, multiplier=1.0, alpha_variance=0.1, xtick_step = 5):
    patch_shape = patch.shape
    unique_labels = np.unique(patch_labels)
    flatten_patch = np.reshape(patch, (patch_shape[0] * patch_shape[1], patch_shape[2]))
    flatten_labels = np.reshape(patch_labels, (patch_shape[0] * patch_shape[1]))

    for label in unique_labels:
        label_indices = np.where(flatten_labels == label)
        label_patch = flatten_patch[label_indices]
        label_patch_variance = np.var(label_patch, axis=0)
        label_patch_mean = np.mean(label_patch, axis=0)

        axis.plot(label_patch_mean)

        indices = np.arange(stop=patch_shape[2])
        axis.fill_between(np.array(indices, dtype=float),
                          y1=label_patch_mean - label_patch_variance * multiplier,
                          y2=label_patch_mean + label_patch_variance * multiplier, alpha=alpha_variance,
                          edgecolor=None)

        mean_plus_variance = label_patch_mean + label_patch_variance * multiplier
        mean_minus_variance = label_patch_mean - label_patch_variance * multiplier

        axis.xaxis.set_ticks(np.arange(0, patch_shape[2] + 1, patch_shape[2] // 4))
        axis.yaxis.set_ticks(np.arange(np.amin(mean_minus_variance) -
                                       (np.amax(mean_plus_variance) - np.amin(mean_minus_variance)) / 100,
                                       np.amax(mean_plus_variance) +
                                       (np.amax(mean_plus_variance) - np.amin(mean_minus_variance)) / 100,
                                       (np.amax(mean_plus_variance) - np.amin(mean_minus_variance)) / 4))
        axis.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))
        axis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))


def render_patch_augmentation(patches):
    fig = plt.figure(figsize=(10, 4))
    num_cols = patches.shape[0]
    title = ['', 'Horizontal flip', 'Vertical flip', 'Rotation (-90º)', 'Rotation (+90º)']
    font, title_font, regular_font = get_plot_fonts()

    for idx, patch in enumerate(patches):
        ax = fig.add_subplot(1, num_cols, idx + 1)
        ax.imshow(get_rgb_chunk(patch), cmap='gray')
        ax.set_title(label=title[idx], **title_font)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

    plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=.1, wspace=0.08)
    plt.savefig(paths.result_folder + "patch_augmentation.png", dpi=300, transparent=True)


def render_patches_examples(original_patches, standard_patches, labels, reduced_labels, target_labels):
    """
    Renders some examples of patches for target labels.
    :return:
    """
    fig = plt.figure(figsize=(13, 8))
    font, title_font, regular_font = get_plot_fonts()

    num_rows = len(target_labels)
    num_cols = 6
    patch_indices = []
    f, axes = plt.subplots(num_rows, num_cols, gridspec_kw={'width_ratios': [7, 7, 1, 9, 1, 9]})

    patch_shape = original_patches[0].shape
    transformed_patch_shape = standard_patches[0].shape

    for idx, label in enumerate(target_labels):
        if idx == 0:
            ax1 = axes[idx, 0]
            ax2 = axes[idx, 1]
            ax2.sharex(axes[0, 0])
        else:
            ax1 = axes[idx, 0]
            ax2 = axes[idx, 1]
            ax1.sharey(axes[0, 0])
            ax2.sharex(axes[0, 0])
            ax2.sharey(axes[0, 1])

        ax3 = axes[idx, 3]
        ax4 = axes[idx, 5]

        patch_idx = get_batch_label(reduced_labels, label)
        patch_indices.append(patch_idx)

        ax1.imshow(get_rgb_chunk(original_patches[patch_idx]), cmap='gray')
        ax2.imshow(get_rgb_mask(labels[patch_idx][:, :], max_labels=np.max(reduced_labels)))
        plot_patch_variance(original_patches[patch_idx], patch_labels=labels[patch_idx], axis=ax3, multiplier=1.0,
                            alpha_variance=0.15, xtick_step=60)
        plot_patch_variance(standard_patches[patch_idx], patch_labels=labels[patch_idx], axis=ax4, multiplier=1.0,
                            alpha_variance=0.15, xtick_step=10)

        ax1.set_ylabel(ylabel="Class " + str(label), **title_font)
        if idx == 0:
            ax1.set_title(label="HSI", **title_font)
            ax2.set_title(label="Mask", **title_font)
            ax3.set_title(label="Original Spectra", **title_font)
            ax4.set_title(label="Transformed Spectra", **title_font)

        ax3.set_xlim([0, patch_shape[2]])
        ax4.set_xlim([0, transformed_patch_shape[2] - 1])

        for ax in [ax1, ax2, axes[idx, 2], axes[idx, 4]]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')

        for null_ax in [axes[idx, 2], axes[idx, 4]]:
            null_ax.spines['top'].set_visible(False)
            null_ax.spines['right'].set_visible(False)
            null_ax.spines['bottom'].set_visible(False)
            null_ax.spines['left'].set_visible(False)

        for axis in [ax3, ax4]:
            for tick_label in axis.get_xticklabels():
                tick_label.set_fontproperties(font)

            for tick_label in axis.get_yticklabels():
                tick_label.set_fontproperties(font)

    plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=.3, wspace=0.3)
    plt.savefig(paths.result_folder + "patch_grid.png", dpi=300, transparent=True)


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
