import config
from cnn_builder import *
from config import *
from dataset_functions import *
from hypercube_set import HypercubeSet
from hypercube_loader import *
import training_history

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
os.chdir(os.getcwd().split("python")[0])

inf = 2e32
network_type = 'allopezr'
read_json_config(paths.config_file, network_type=network_type)

#### Hypercube reading
hc_set = HypercubeSet(hc_array=load_hypercubes(plot_hc=False, plot_mask=True, n_max_cubes=inf))
hc_set.print_metadata()

#### Dataset creation
hc_set.obtain_ground_labels()
hc_set.obtain_train_indices(test_percentage=test_split, patch_size=config.patch_size,
                            patch_overlapping=config.patch_size-1)

#### Build network
network_name = get_name(network_type)
num_classes = hc_set.get_num_classes()
img_shape = (config.patch_size, config.patch_size, config.num_target_features)

history = training_history.TrainingHistory.load(network_name)
model = load_model(network_type, network_name, num_classes, img_shape)

callbacks, time_callback = get_callback_list(model_name=network_name)

#### Preprocessing
hc_set.standardize()

### Split test
X_test, y_test = hc_set.split_test(patch_size=config.patch_size)
y_test = reduce_labels_center(y_test)
(X_test, y_test), _, _ = balance_classes(X_test, y_test, reduce=True, clustering=False)

print("Test set size: " + str(len(X_test)))
print(X_test.shape)
print(y_test.shape)

### Results
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss: " + str(test_loss) + ", Test Accuracy: " + str(test_accuracy))

render_model_history(history, model_name=network_name)

prediction = model.predict(X_test)
model_predictions = np.argmax(prediction, axis=1)
render_confusion_matrix(y_test, model_predictions)