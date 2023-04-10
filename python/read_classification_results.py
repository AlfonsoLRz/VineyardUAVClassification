from cnn_builder import *
import papers.aspn
import glob
import keras
import os
import paths
import pickle
from rendering import *

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
os.chdir(os.getcwd().split("python")[0])

network_type = 'allopezr_2d'
read_json_config(paths.config_file, network_type=network_type)
network_name = get_name(network_type)

# Look for a file with the network name under the history folder
history_file = os.path.join(paths.result_folder + 'history/', network_name + "*.p")
history_files = glob.glob(history_file)

for history_file in history_files:
    # Load history with pickle
    history = pickle.load(open(history_file, "rb"))
    render_model_history(history, model_name=network_name)

# Read network weights
network_weights_file = os.path.join(paths.result_folder + 'network/', network_name + "*.h5")
network_weights_files = glob.glob(network_weights_file)
if len(network_weights_files) > 0:
    network_weights_file = network_weights_files[-1]
    print("Loading weights file " + network_weights_file)
    model = keras.models.load_model(network_weights_file,
                                    custom_objects={'SpatialAttention': papers.aspn.SpatialAttention,
                                                    'SecondOrderPooling': papers.aspn.SecondOrderPooling})

    model.summary()

# Read the metrics
metrics_file = os.path.join(paths.result_folder + 'metrics/', network_name + ".p")
metrics_files = glob.glob(metrics_file)
if len(metrics_files) > 0:
    metrics_file = metrics_files[-1]
    print("Loading metrics file " + metrics_file)
    metrics = pickle.load(open(metrics_file, "rb"))

    # Plot the metrics
    metrics.print_metrics()

