import config
import paths
from cnn_builder import *
from config import *
from dataset_functions import *
import gc
from hypercube_set import HypercubeSet
from hypercube_loader import *
import numpy as np
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

inf = 2e32
max_samples = 50000
sampling_strategy = 'not minority'
if paths.target_area == 2:
    sampling_strategy = 'majority'

network_type = 'allopezr_2d'
read_json_config(paths.config_file, network_type=network_type)

for patch_size in np.arange(31, 33, 2):
    config.patch_size = patch_size
    config.patch_overlapping = patch_size - 1

    #### Hypercube reading
    hc_set = HypercubeSet(hc_array=load_hypercubes(plot_hc=False, plot_mask=True, n_max_cubes=inf))
    hc_set.print_metadata()

    #### Dataset creation
    hc_set.obtain_ground_labels()
    hc_set.obtain_train_indices(test_percentage=test_split, patch_size=config.patch_size,
                                patch_overlapping=config.patch_overlapping)

    #### Preprocessing
    hc_set.standardize(num_features=config.num_target_features,
                       selection_method=LayerSelectionMethod.FACTOR_ANALYSIS)

    #### Build network
    metrics = training_metrics.TrainingMetrics()
    num_classes = hc_set.get_num_classes()
    img_shape = (patch_size, patch_size, config.num_target_features)

    network_name = get_name(network_type) + '_window_size_test'
    model = build_network(network_type=network_type, num_classes=num_classes, image_dim=img_shape)
    compile_network(model, network_type, network_name, num_classes, show_summary=True, render_image=True)
    starting_index = 0

    ### Split test
    X_test, y_test = hc_set.split_test(patch_size=patch_size)
    y_test = reduce_labels_center(y_test)
    (X_test, y_test), _, _ = balance_classes(X_test, y_test, reduce=True, clustering=False,
                                             strategy=sampling_strategy)

    history = training_history.TrainingHistory(accuracy_name='sparse_categorical_accuracy')
    callbacks, time_callback = get_callback_list(model_name=network_name)

    #### Training
    while True:
        X_train, y_train = hc_set.split_train(patch_size=patch_size, max_train_samples=max_samples,
                                              starting_index=starting_index, remove=False)

        if len(X_train) > 0:
            y_train = reduce_labels_center(y_train)
            (patch, patch_label), rest_patch, rest_label = balance_classes(X_train, y_train, reduce=True,
                                                                           clustering=False, strategy=sampling_strategy)

            X_train, y_train = [], []
            X_train.append(patch)
            y_train.append(patch_label)

            for i in range(len(X_train)):
                X_train_augment, y_train_augmented = augment_chunks(X_train[i], y_train[i])
                history.append_history(run_model(model, X_train_augment, y_train_augmented,
                                                 validation_split=validation_split, callbacks=callbacks).history,
                                       training_callback=time_callback, samples=X_train_augment)

                del X_train_augment, y_train_augmented

            del patch, patch_label, rest_patch, rest_label
        else:
            break

        del X_train, y_train
        starting_index += max_samples

        gc.collect()

    test_prediction_prob = model.predict(X_test)
    test_prediction = np.argmax(test_prediction_prob, axis=1)
    metrics.append_evaluation(y_test, test_prediction, test_prediction_prob)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Loss: " + str(test_loss) + ", Test Accuracy: " + str(test_accuracy))

    history.save(network_name)

    # Graphic results
    render_model_history(history, model_name=network_name)
    render_confusion_matrix(y_test, test_prediction)

    del model
    del hc_set
    del X_test, y_test
    gc.collect()

    metrics.print_metrics()
    metrics.save(network_name)
