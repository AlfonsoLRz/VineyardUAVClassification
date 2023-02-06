import config
from cnn_builder import *
from config import *
from dataset_functions import *
from hypercube_set import HypercubeSet
from hypercube_loader import *
import numpy as np
import rendering

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
os.chdir(os.getcwd().split("python")[0])

inf = 2e32
max_samples = 100000

#force_gpu()
read_json_config(paths.config_file, network_type=None)

#### Hypercube reading
hc_set = HypercubeSet(hc_array=load_hypercubes(plot_hc=False, plot_mask=True, n_max_cubes=inf))
hc_set.print_metadata()

#### Dataset creation
hc_set.obtain_ground_labels()
hc_set.obtain_train_indices(test_percentage=test_split, patch_size=config.patch_size,
                            patch_overlapping=config.patch_size-1)

#### Build network
network_type = 'allopezr_2d'
read_json_config(paths.config_file, network_type=network_type)

network_name = get_name(network_type)
num_classes = hc_set.get_num_classes()
img_shape = (config.patch_size, config.patch_size, config.num_target_features)

model = build_network(network_type=network_type, num_classes=num_classes, image_dim=img_shape)
compile_network(model, network_type, network_name, num_classes, show_summary=True, render_image=True)

callbacks = get_callback_list(model_name=network_name)

#### Preprocessing
hc_set.standardize()

### Split test
X_test, y_test = hc_set.split_test(patch_size=patch_size)
y_test = reduce_labels_center(y_test)
(X_test, y_test), _, _ = balance_classes(X_test, y_test, reduce=True, clustering=False)

#### Training
while True:
    X_train, y_train = hc_set.split_train(patch_size=config.patch_size, max_train_samples=max_samples)

    if len(X_train) > 0:
        y_train = reduce_labels_center(y_train)
        (patch, patch_label), rest_patch, rest_label = balance_classes(X_train, y_train, reduce=True, clustering=False)
        (rest_patch, rest_label), _, _ = balance_classes(rest_patch, rest_label, reduce=True, clustering=False)

        X_train, y_train = [], []
        X_train.append(patch)
        X_train.append(rest_patch)
        y_train.append(patch_label)
        y_train.append(rest_label)

        for i in range(len(X_train)):
            print("Training on {} samples".format(len(X_train[i])))
            X_train_augment, y_train_augmented = augment_chunks(X_train[i], y_train[i])
            print("Augmented to {} samples".format(len(X_train_augment)))
            history = run_model(model, X_train_augment, y_train_augmented, validation_split=validation_split,
                                callbacks=callbacks)
    else:
        break
