from cnn_builder import *
from config import *
from dataset_functions import *
from hypercube_set import HypercubeSet
from hypercube_loader import *
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
os.chdir(os.getcwd().split("python")[0])

inf = 2e32

force_gpu()

hc_set = HypercubeSet(hc_array=load_hypercubes(plot_hc=False, plot_mask=False, n_max_cubes=inf))
hc_set.print_metadata()

hc_set.obtain_ground_labels()
hc_set.obtain_train_indices(test_percentage=test_split, patch_size=patch_size, patch_overlapping=patch_overlapping)
hc_set.standardize()

X_train, y_train = hc_set.split_train(patch_size=patch_size)
y_train = reduce_labels_center(y_train)

X_test, y_test = hc_set.split_test(patch_size=patch_size)
y_test = reduce_labels_center(y_test)

render_mask_histogram(y_train)
render_mask_histogram(y_test)

(patch, patch_label), rest_patch, rest_label = balance_classes(X_train, y_train, reduce=True, clustering=False)
X_train, y_train = [], []
X_train.append(patch)
X_train.append(rest_patch)
y_train.append(patch_label)
y_train.append(rest_label)
vegetation_indices = np.where(y_train[1] == 0)
X_train[1] = np.delete(X_train[1], vegetation_indices)
y_train[1] = np.delete(y_train[1], vegetation_indices)

X_train_flatten = get_center(X_train[0])
X_train_augment, y_train_augmented = augment_chunks(X_train[0], y_train[0])

network_type = 'allopezr_2d'
#read_json_config(paths.config_file, network_type=network_type)

network_name = get_name(network_type)
num_classes = hc_set.get_num_classes()
img_shape = X_train[0][0].shape

model = build_network(network_type=network_type, num_classes=num_classes, image_dim=img_shape)
compile_network(model, network_type, network_name, num_classes, show_summary=True, render_image=True)
history = run_model(model, X_train_augment, y_train_augmented, validation_split=validation_split, callbacks=get_callback_list(model_name=network_name))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss: " + str(test_loss) + ", Test Accuracy: " + str(test_accuracy))

render_model_history(history, model_name=network_name, accuracy="sparse_categorical_accuracy")
