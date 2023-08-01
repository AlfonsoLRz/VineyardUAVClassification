import copy
from hypercube_set import *
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import paths
import plotly.io as plt_io
import plotly.graph_objects as go
import plotly.offline
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication
from randomness import *
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import sys
import training_history
import umap.umap_ as umap


def get_plot_fonts():
    font = 'Adobe Devanagari'
    title_font = {'fontname': font, 'size': 15}
    regular_font = {'fontname': font, 'size': 14}
    font = font_manager.FontProperties(family=font, size=14)

    return font, title_font, regular_font


def render_confusion_matrix(y_test, y_pred, model_name):
    """
    Renders the confusion matrix of the model predictions.
    """
    font, title_font, regular_font = get_plot_fonts()

    flatten_y_test = np.reshape(y_test, (-1,))
    flatten_y_pred = np.reshape(y_pred, (-1,))

    cm = confusion_matrix(flatten_y_test, flatten_y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # sn.set(font_scale=1)
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='g')
    # sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues')
    plt.ylabel('Actual', **regular_font)
    plt.xlabel('Predicted', **regular_font)
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font)

    plt.tight_layout()
    plt.savefig(paths.result_folder + 'images/' + model_name + '_confusion_matrix.png')
    plt.show(block=False)


def render_hc_spectrum_label(hc_numpy, mask):
    """
    Renders the spectrum of every label in the hypercube.
    """
    n_classes = np.unique(mask)
    font, title_font, regular_font = get_plot_fonts()
    rows, cols = 2, len(n_classes) // 2

    dict = red_vineyard_name
    width = 16
    if paths.target_area == 'white':
        cols += 1
        width += 4
        dict = white_vineyard_name

    plt.subplots(figsize=(width, 6))
    for i, class_id in enumerate(n_classes):
        row = i // cols
        col = i % cols

        plt.subplot(rows, cols, 1 + i)
        sample_subset = hc_numpy[mask == class_id, :]

        mean = np.average(sample_subset, axis=0)
        variance = np.var(sample_subset, axis=0)
        plt.fill_between(range(len(mean)), mean - variance, mean + variance, alpha=0.2)
        plt.plot(mean, label=class_id)

        # Change font of axes
        plt.title(dict[class_id], **title_font)
        if row > 0:
            plt.xlabel('Spectral band', **regular_font)
        if col == 0:
            plt.ylabel('Reflectance', **regular_font)
        plt.xticks(fontproperties=font)
        plt.yticks(fontproperties=font)

    plt.tight_layout()
    plt.savefig(paths.result_folder + 'images/hc_spectrum_label.png', dpi=500)
    plt.show()


def render_label_diff(label_diff, filename, dpi=500):
    """
    Renders the difference between the ground truth and the predicted labels.
    """
    # Get pixels from hypercube zero whose labels are different from the ground truth

    plt.imshow(label_diff, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
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


def render_mask_histogram(label, render_classes_count=2):
    """
    Renders the histogram of the mask.
    """
    unique, counts = np.unique(label, return_counts=True)

    # Plot
    plt.figure(figsize=(7, 3))
    plt.bar(unique, counts, width=0.5)
    plt.xticks(unique)

    # Plot number of samples of top two classes as lines
    if render_classes_count > 0:
        top_classes = counts.argsort()[-render_classes_count:][::-1]
        for i in top_classes:
            plt.plot([0 - 0.5, np.max(unique) + 1 - 0.5], [counts[i], counts[i]], 'r--', linewidth=1)

    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.xlim([0 - 0.5, np.max(unique) + 1 - 0.5])
    plt.tight_layout()
    # plt.savefig(paths.result_folder + 'images/mask_histogram.png', dpi=500)
    plt.show()


def render_model_history(history, model_name):
    """
    Renders the history of the model after training.
    """
    font, title_font, regular_font = get_plot_fonts()

    h_epochs = range(1, history.get_history_length() + 1)
    accuracy = history.get_accuracy_key()
    history_vector = history.get_history()

    plt.plot(h_epochs, history_vector[accuracy], label="Training accuracy")
    plt.plot(h_epochs, history_vector['val_' + accuracy], label="Validation accuracy")
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(font)

    plt.title('a) Training and validation accuracy', fontdict=title_font)
    plt.legend(prop=font, frameon=False)
    plt.ylabel('Accuracy', **regular_font)
    plt.xlabel('Epoch', **regular_font)
    plt.tight_layout()
    plt.savefig(paths.result_folder + 'images/' + model_name + '_accuracy.png')
    plt.figure()

    # Loss
    plt.plot(h_epochs, history_vector['loss'], label="Training loss")
    plt.plot(h_epochs, history_vector['val_loss'], label="Validation loss")
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(font)

    plt.title('b) Training and validation loss', fontdict=title_font)
    plt.legend(prop=font, frameon=False)
    plt.ylabel('Loss', **regular_font)
    plt.xlabel('Epoch', **regular_font)
    plt.tight_layout()
    plt.savefig(paths.result_folder + 'images/' + model_name + '_loss.png')
    plt.show()


def render_network_training(network_labels, training_time, num_params, title=None, bar_width=0.5):
    font, title_font, regular_font = get_plot_fonts()

    fig = plt.figure(figsize=(10, 10 / 1.8))
    ax = fig.add_subplot(111)
    response_time_y = [x / 60.0 for x in training_time]
    params_y = [x for x in num_params]

    data = np.concatenate((np.array([['network', 'time', 'params']]),
                           np.array([network_labels, response_time_y, params_y]).T), axis=0)
    pd_df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:]).astype(float)
    pd_df.plot(kind='bar', secondary_y='params', ax=ax, width=bar_width)

    axes = fig.axes
    for ax in axes:
        set_axis_font(ax, font)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.xlabel('Network', fontdict=regular_font)
    axes[0].set_ylabel('Training time (minutes)', fontdict=regular_font)
    axes[1].set_ylabel('#Parameters', fontdict=regular_font)
    # Set rotation of x-axis labels
    for item in axes[0].get_xticklabels():
        item.set_rotation(45)
    if title is not None:
        plt.title(title, fontdict=title_font)

    cyan_patch = mpatches.Patch(color='cyan', label='Training time')
    orange_patch = mpatches.Patch(color='orange', label='Number of parameters')
    plt.legend(handles=[cyan_patch, orange_patch], prop=font, frameon=False, loc='upper left')
    plt.tight_layout()
    plt.savefig(paths.result_folder + 'network_training.png')
    plt.show()


def render_time_capacity(response_time, capacity, title=None, bar_width=0.6):
    font, title_font, regular_font = get_plot_fonts()

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    x_val = [x[0] for x in response_time]
    response_time_y = [x[1] / 60.0 for x in response_time]
    capacity_y = [x[1] for x in capacity]

    data = np.concatenate((np.array([['patchsize', 'time', 'capacity']]),
                           np.array([x_val, response_time_y, capacity_y]).T), axis=0)
    pd_df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0].astype(np.float).astype(np.int), columns=data[0, 1:]) \
        .astype(float)
    pd_df.plot(kind='bar', secondary_y='capacity', ax=ax, width=bar_width)

    axes = fig.axes
    for ax in axes:
        set_axis_font(ax, font)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.xlabel('Window size', fontdict=regular_font)
    axes[0].set_ylabel('Training time (minutes)', fontdict=regular_font)
    axes[1].set_ylabel('#Parameters', fontdict=regular_font)
    if title is not None:
        plt.title(title, fontdict=title_font)
    plt.tight_layout()
    plt.savefig(paths.result_folder + 'time_capacity.png', dpi=300)
    plt.show()


def render_transformation_grid(patches, transformer, x, y, z=0):
    transformed_patches = patches.copy()

    for i in range(len(patches)):
        transformed_patches[i, :, :, z] = transformer(image=patches[i, :, :, z])["image"]

    fig = plt.figure(figsize=(x * 2, y * 2))

    for i in range(x * y // 2):
        # Random image
        idx = np.random.randint(0, len(patches))

        ax = fig.add_subplot(y, x, i * 2 + 1, xticks=[], yticks=[])
        if i // (x // 2) == 0:
            ax.set_title('Original', fontdict={'fontsize': 10})
        ax.imshow(patches[idx, :, :, z])

        ax = fig.add_subplot(y, x, i * 2 + 2, xticks=[], yticks=[])
        if i // (x // 2) == 0:
            ax.set_title('Transformed', fontdict={'fontsize': 10})
        ax.imshow(transformed_patches[idx, :, :, z])

    plt.tight_layout()
    plt.savefig(paths.result_folder + 'transformation_grid.png')
    plt.show()


def render_window_size_metric(patch_size_metric, annotate_indices=[], title=None):
    """
    Renders the metric of the patch size.
    """
    font, title_font, regular_font = get_plot_fonts()

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    x_val = [x[0] for x in patch_size_metric]
    y_val = [x[1] for x in patch_size_metric]
    ax.plot(x_val, y_val, 'rs', x_val, y_val, 'r-')
    ax.set_xlim([np.min(x_val) - 0.6, np.max(x_val) + 0.6])
    ax.set_ylim([np.min(y_val) - 0.02, np.max(y_val) + 0.02])
    ax.set_xticks(x_val)
    set_axis_font(ax, font)

    # Annotate only certain values
    for idx in annotate_indices:
        ax.annotate('{0:.4f}'.format(y_val[idx][0]), xy=(x_val[idx], y_val[idx][0]),
                    xytext=(x_val[idx] - 1, y_val[idx][0] + 0.012), **regular_font)

    plt.xlabel('Patch size', fontdict=regular_font)
    plt.ylabel('Overall Accuracy', fontdict=regular_font)
    if title is not None:
        plt.title(title, fontdict=title_font)
    plt.tight_layout()
    plt.savefig(paths.result_folder + 'window_size_test.png', dpi=300)
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


def plot_patch_variance(patch, patch_labels, axis, multiplier=1.0, alpha_variance=0.1, xtick_step=5):
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

        set_axis_font(ax3, font)
        set_axis_font(ax4, font)

    plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=.3, wspace=0.3)
    plt.savefig(paths.result_folder + "patch_grid.png", dpi=300, transparent=True)


def render_manifold_separability(embedding, labels, include_annotations=True):
    """
    Renders the t-SNE of the spectrum of every label in the hypercube.
    """
    different_labels = np.unique(labels)
    num_different_labels = len(different_labels)

    # Adjust size of plot
    font, title_font, regular_font = get_plot_fonts()
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
    if include_annotations:
        ax = plt.gca()
        annotations = copy.copy(labels)
        annotations = [str(x) for x in annotations]

        # annotations = get_annotation_labels(embedding, labels)
        #
        # for idx in range(len(annotations)):
        #     ax.annotate(annotations[idx], (embedding[idx, 0], embedding[idx, 1]),
        #                 xytext=(embedding[idx, 0] + 0.05, embedding[idx, 1] + 0.3),
        #                 bbox=dict(boxstyle="round", alpha=0.4), **regular_font)
        for label in different_labels:
            # Pick one random sample
            label_idx = np.random.choice(np.where(labels == label)[0])
            ax.annotate(annotations[label_idx], (embedding[label_idx, 0], embedding[label_idx, 1]),
                        xytext=(embedding[label_idx, 0] + 0.05, embedding[label_idx, 1] + 0.3),
                        bbox=dict(boxstyle="round", alpha=0.2), **regular_font)

    plt.gca().set_aspect('equal', 'datalim')
    cb = plt.colorbar(boundaries=np.arange(num_different_labels + 1) - 0.5)
    cb.set_ticks(np.arange(num_different_labels))
    for t in cb.ax.get_yticklabels():
        t.set_fontproperties(font)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(paths.result_folder + "separability.png", dpi=500, transparent=True)
    plt.show()


def render_3d_manifold_separability(embedding, label):
    """
    Renders the 3D unmixed manifold of data.
    """
    annotation_labels = copy.copy(label)
    annotation_labels = [str(x) for x in annotation_labels]
    font = 'Adobe Devanagari'

    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=label,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=1,
            line_width=1,
            colorbar=dict(
                title="Red Variety"
            ),
        ),
    )])
    # tight layout
    ann = [dict(x=x, y=y, z=z, text=annotation, showarrow=False) for x, y, z, annotation in
           zip(embedding[:, 0], embedding[:, 1], embedding[:, 2], annotation_labels)]
    # fig.update_layout(
    #     scene=dict(
    #         annotations=ann
    #     )
    # )
    # fig.update_layout(margin=dict(l=50, r=50, b=50, t=50), width=1080, height=975)
    fig.layout.template = 'plotly'
    fig.update_layout(
        font_family=font,
        font_size=14,
        title_font_family=font,
        width=1200,
        height=800
    )
    fig.update_xaxes(title_font_family=font)
    fig.update_layout(showlegend=False)

    config = {
        'toImageButtonOptions': {
            'format': 'png',  # one of png, svg, jpeg, webp
            'filename': 'D:/Test',
            'scale': 6  # Multiply title/legend/axis/canvas sizes by this factor
        }
    }

    # fig.show(config=config)
    show_in_window(fig)


def set_axis_font(axis, font):
    for tick_label in axis.get_xticklabels():
        tick_label.set_fontproperties(font)

    for tick_label in axis.get_yticklabels():
        tick_label.set_fontproperties(font)


def show_in_window(fig):
    filename = "Components.html"
    plotly.offline.plot(fig, filename=filename, auto_open=False)

    app = QApplication(sys.argv)
    web = QWebEngineView()
    file_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), filename))
    web.load(QUrl.fromLocalFile(file_path))
    web.show()
    sys.exit(app.exec_())


## Manifold anotations

def get_annotation_labels(embedding, labels, prob=0.05):
    annotation_labels = copy.copy(labels)
    annotation_labels = [str(x) for x in annotation_labels]

    regular_grid_lod_x, regular_grid_lod_y = 5, 10
    rect_size_x, rect_size_y = 0.3, 0.2

    x_min, x_max = np.min(embedding[:, 0]) - 0.0001, np.max(embedding[:, 0]) + 0.0001
    y_min, y_max = np.min(embedding[:, 1]) - 0.0001, np.max(embedding[:, 1]) + 0.0001

    x_size, y_size = (x_max - x_min) / regular_grid_lod_x, (y_max - y_min) / regular_grid_lod_y
    regular_grid = np.zeros(shape=(int(np.ceil(((x_max - x_min) / x_size))), int(np.ceil((y_max - y_min) / y_size))))
    included_labels = []

    # Fill borders
    regular_grid[0, :] = 1
    regular_grid[-1, :] = 1
    regular_grid[:, 0] = 1
    regular_grid[:, -1] = 1

    for idx, label in enumerate(annotation_labels):
        if np.random.rand() < prob and not label_overlaps(embedding[idx, 0], embedding[idx, 1], rect_size_x,
                                                          rect_size_y, included_labels):
            included_labels.append((embedding[idx, 0], embedding[idx, 1]))
            mark_grid(embedding[idx, 0], embedding[idx, 1], x_min, x_size, y_min, y_size, regular_grid)
        else:
            annotation_labels[idx] = ""

    return annotation_labels


def map_x_y(x, y, x_min, x_size, y_min, y_size):
    return int(np.floor((x - x_min) / x_size)), int(np.floor((y - y_min) / y_size))


def is_regular_grid_occupied(x, y, x_min, x_size, y_min, y_size, regular_grid):
    x_d, y_d = map_x_y(x, y, x_min, x_size, y_min, y_size)
    return regular_grid[x_d, y_d] > 0


def mark_grid(x, y, x_min, x_size, y_min, y_size, regular_grid):
    x_d, y_d = map_x_y(x, y, x_min, x_size, y_min, y_size)
    regular_grid[x_d, y_d] += 1


def label_overlaps(x, y, rect_size_x, rect_size_y, included_labels):
    xmin2, xmax2 = x - rect_size_x, x + rect_size_x
    ymin2, ymax2 = y - rect_size_y, y + rect_size_y

    for (x_1, y_1) in included_labels:
        xmin1, xmax1 = x_1 - rect_size_x, x_1 + rect_size_x
        ymin1, ymax1 = y_1 - rect_size_y, y_1 + rect_size_y

        if xmax1 >= xmin2 and xmax2 >= xmin1 and ymax1 >= ymin2 and ymax2 >= ymin1:
            return True

    return False
