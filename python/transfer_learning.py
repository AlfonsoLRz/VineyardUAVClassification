import config
import paths
from cnn_builder import *
from config import *
from dataset_functions import *
import gc
from hypercube_set import HypercubeSet
from hypercube_loader import *
import papers
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
max_samples = 100000
num_tests = 1
sampling_strategy = 'not minority'

# Experiment setup
trainable_layers = [
    np.asarray([np.arange(0, 33, 1)]),
    np.asarray([np.arange(21, 33, 1)]),
    np.asarray([np.arange(11, 33, 1)]),
    np.append(np.arange(2, 8, 1), np.arange(32, 33, 1))
]

network_type = 'allopezr_2d'
read_json_config(paths.config_file, network_type=network_type)
network_name = get_name(network_type)

#### Hypercube reading
hc_set = HypercubeSet(hc_array=load_hypercubes(plot_hc=False, plot_mask=True, n_max_cubes=inf))

#### Dataset creation
hc_set.obtain_ground_labels()
hc_set.obtain_train_indices(test_percentage=test_split, patch_size=config.patch_size,
                            patch_overlapping=config.patch_overlapping)

# Remove unwanted labels
num_classes = hc_set.get_num_classes()
hc_set.swap_classes(0, num_classes - 1)
hc_set.print_metadata()

#### Preprocessing
hc_set.standardize(num_features=config.num_target_features, selection_method=LayerSelectionMethod.FACTOR_ANALYSIS)

#### Build network
num_classes = hc_set.get_num_classes()
num_classes -= 1
img_shape = (config.patch_size, config.patch_size, config.num_target_features)

### Split test
X_test, y_test = hc_set.split_test(patch_size=config.patch_size)
y_test = reduce_labels_center(y_test)
X_test, y_test = remove_labels(X_test, y_test, [num_classes])
#(X_test, y_test), _, _ = balance_classes(X_test, y_test, reduce=True, clustering=False, strategy=sampling_strategy)

for experiment_id in range(len(trainable_layers)):
    metrics = training_metrics.TrainingMetrics()

    network_name = get_name(network_type)
    network_file = os.path.join('results/umat/network/', network_name + "_indian_pines_0.h5")
    model = keras.models.load_model(network_file, custom_objects={'SpatialAttention': papers.aspn.SpatialAttention,
                                                                  'SecondOrderPooling': papers.aspn.SecondOrderPooling})

    x = Dense(num_classes, activation=config.last_activation)(model.layers[-2].output)
    model = Model(inputs=model.input, outputs=x)
    model.trainable = True
    for (idx, layer) in enumerate(model.layers):
        if idx not in trainable_layers[experiment_id]:
            model.layers[idx].trainable = False

    network_name = network_name + '_exp' + str(experiment_id)

    compile_network(model, network_type, network_name, num_classes, show_summary=True, render_image=True)
    model.save_weights(network_name + "_init.h5")

    for i in range(num_tests):
        print("Test " + str(i+1) + "/" + str(num_tests))
        print("---------------------------------")

        ### Restore weights
        model.load_weights(network_name + "_init.h5")
        starting_index = 0

        history = training_history.TrainingHistory(accuracy_name='sparse_categorical_accuracy')
        callbacks, time_callback = get_callback_list(model_name=network_name, test_id=i)

        #### Training
        while True:
            X_train, y_train = hc_set.split_train(patch_size=config.patch_size, max_train_samples=max_samples,
                                                  starting_index=starting_index, remove=False)

            if len(X_train) > 0:
                y_train = reduce_labels_center(y_train)
                X_train, y_train = remove_labels(X_train, y_train, [num_classes])

                (patch, patch_label), _, _ = balance_classes(X_train, y_train, reduce=True,
                                                             clustering=False, strategy=sampling_strategy)
                #(rest_patch, rest_label), _, _ = balance_classes(rest_patch, rest_label, reduce=True, clustering=False)
                render_mask_histogram(patch_label)

                X_train, y_train = [], []
                X_train.append(patch)
                y_train.append(patch_label)

                for i in range(len(X_train)):
                    X_train_augment, y_train_augmented = augment_chunks(X_train[i], y_train[i])
                    history.append_history(run_model(model, X_train_augment, y_train_augmented,
                                                     validation_split=validation_split, callbacks=callbacks).history,
                                           training_callback=time_callback, samples=X_train_augment)

                    del X_train_augment, y_train_augmented

                del patch, patch_label
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

        history.save(network_name, test_id=i)

        # Graphic results
        render_model_history(history, model_name=network_name)
        render_confusion_matrix(y_test, test_prediction, model_name=network_name)

    metrics.print_metrics()
    metrics.save(network_name)

    # Delete weight file
    os.remove(network_name + "_init.h5")

    del model
