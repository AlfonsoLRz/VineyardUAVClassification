from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate, VerticalFlip, Crop, PixelDropout, CropAndPad, RandomBrightnessContrast
)
import config
import paths
import randomness
from cnn_builder import *
from config import *
from dataset_functions import *
from functools import partial
import gc
from hypercube_set import HypercubeSet
from hypercube_loader import *
import numpy as np
import random
import randomness
import rendering
import training_history
import training_metrics

import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
os.chdir(os.getcwd().split("python")[0])

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

font_mapping = {'family': 'Palatino Linotype', 'weight': 'normal', 'size': 11}
plt.rc('font', **font_mapping)

inf = 2e32
network_type = 'allopezr_2d'
read_json_config(paths.config_file, network_type=network_type)

transforms = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Rotate(limit=(-360, -360), p=0.1),
    Rotate(limit=(-270, -270), p=0.1),
    Rotate(limit=(-180, -180), p=0.1),
    Rotate(limit=(-90, -90), p=0.1),
    Rotate(limit=(90, 90), p=0.1),
    Rotate(limit=(180, 180), p=0.1),
    Rotate(limit=(270, 270), p=0.1)
])

hc_array_red, max_class_idx, _ = load_hypercubes(plot_hc=False, plot_mask=False, n_max_cubes=inf,
                                                 folder='media/Mateus_2022/red/')
hc_array_white, max_class_idx, _ = load_hypercubes(plot_hc=False, plot_mask=False, n_max_cubes=inf,
                                                   folder='media/Mateus_2022/white/', baseline_class_idx=max_class_idx)
hc_array = hc_array_red + hc_array_white
hc_set = HypercubeSet(hc_array)
hc_set.identify_ground_samples()
hc_set.split_hypercubes(test_percentage=test_split)
hc_set.standardize(num_features=config.num_target_features, selection_method=LayerSelectionMethod.FACTOR_ANALYSIS)

for patch_size in np.arange(5, 32, 2):
    metrics = training_metrics.TrainingMetrics()
    config.patch_size = patch_size
    config.patch_overlapping = config.patch_size - 1
    num_training_splits = int((config.patch_size - 5) / (31 - 5) * 8 + 1)
    percentage_step = 1.0 / num_training_splits
    num_iterations = int(config.epochs / num_training_splits / config.num_transformation_iterations)

    print(num_training_splits)

    num_classes = hc_set.get_num_classes() - 1
    img_shape = (config.patch_size, config.patch_size, config.num_target_features)

    #### Build network
    network_name = get_name(network_type) + '_window_size_test'
    model = build_network(network_type=network_type, num_classes=num_classes, image_dim=img_shape)
    compile_network(model, network_type, network_name, num_classes, show_summary=True, render_image=True)

    history = training_history.TrainingHistory(accuracy_name='sparse_categorical_accuracy')
    callbacks, time_callback = get_callback_list(model_name=network_name)

    ### Training
    for batch in range(num_training_splits):
        X_train, y_train = hc_set.split(patch_size=config.patch_size, patch_overlap=config.patch_overlapping,
                                        train=True, start_percentage=percentage_step * batch,
                                        end_percentage=percentage_step * (batch + 1))

        print(X_train.shape)

        X_train, y_train = randomness.stratified_sampling(X_train, y_train, use_float=True, num_reduced_classes=3)
        render_mask_histogram(y_train)

        for it in range(config.num_transformation_iterations):
            X_train_transformed = X_train.copy()

            for i in range(len(X_train_transformed)):
                random_seed = np.random.randint(0, X_train_transformed.shape[0] * 10)
                for layer in range(X_train_transformed.shape[-1]):
                    randomness.set_seed(random_seed)
                    X_train_transformed[i, :, :, layer] = transforms(image=X_train_transformed[i, :, :, layer])["image"]

            X_train_it, X_validation_it, y_train_it, y_validation_it = split_train_test(X_train_transformed, y_train,
                                                                                        test_size=config.validation_split * (
                                                                                                1.0 - config.test_split),
                                                                                        random_seed=randomness.random_seed)

            del X_train_transformed
            gc.collect()

            history.append_history(
                run_model(model, X_train_it, y_train_it, X_validation_it, y_validation_it, callbacks=callbacks,
                          num_epochs=num_iterations).history, time_callback)

            del X_train_it, X_validation_it, y_train_it, y_validation_it
            gc.collect()

        del X_train, y_train
        gc.collect()

    #### Split test into batches
    y_test_global = []
    test_prediction_global = []
    test_prediction_prob_global = []
    percentage_step = 1.0 / config.num_test_splits

    for batch in range(config.num_test_splits):
        X_test, y_test = hc_set.split(patch_size=config.patch_size, patch_overlap=config.patch_overlapping,
                                      train=False, start_percentage=percentage_step * batch,
                                      end_percentage=percentage_step * (batch + 1))
        test_prediction_prob = model.predict(X_test)
        test_prediction = np.argmax(test_prediction_prob, axis=1)

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print("Test Loss: " + str(test_loss) + ", Test Accuracy: " + str(test_accuracy))

        # y test to int
        y_test_global.extend(np.asarray(y_test, dtype="int64").tolist())
        test_prediction_global.extend(test_prediction.tolist())
        test_prediction_prob_global.extend(test_prediction_prob.tolist())

        del X_test, y_test, test_prediction, test_prediction_prob
        gc.collect()

    y_test_global = np.asarray(y_test_global)
    test_prediction_global = np.asarray(test_prediction_global)
    test_prediction_prob_global = np.asarray(test_prediction_prob_global)

    metrics.append_evaluation(y_test_global, test_prediction_global, test_prediction_prob_global)
    history.save(network_name)

    # Graphic results
    render_model_history(history, model_name=network_name)
    render_confusion_matrix(y_test_global, test_prediction_global, model_name=network_name)

    metrics.print_metrics()
    metrics.save(network_name)
