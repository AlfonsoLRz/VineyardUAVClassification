import config as cfg
import ctypes
import json
import paths
import randomness
from config import *
from papers.allopezr_2d import *
from papers.allopezr_3d import *
from papers.aspn import *
from papers.fsk_net import *
from papers.hybrid_sn import *
from papers.jigsaw_hsi import *
from papers.spectral_net import *
from papers.lt_cnn import *
from papers.nezami import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib

# -------------- SUPPORT FUNCTIONS --------------


def force_gpu():
    tf.compat.v1.Session()
    ctypes.WinDLL("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDNN/v8.6.0/bin/cudnn64_8.dll")
    tf.debugging.set_log_device_placement(True)


def get_callback_list(model_name, monitor_early_stopping='sparse_categorical_accuracy', patience=10):
    return [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.EarlyStopping(
            monitor=monitor_early_stopping,
            patience=patience,
            verbose=3,
            mode='auto'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=paths.result_folder + 'network/' + model_name + '.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        TimeCallback(),
    ]


def get_metrics(num_classes):
    return [
        keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy'),
    ]


def get_name(network_type):
    return network_type + '_' + str(patch_size) + 'x' + str(patch_overlapping) + '_' + \
           str(training_config[network_type]['start_size'])


# --------------- MODELS ----------------

def get_allopezr_2d(config, img_size, num_classes):
    model = get_allopezr_2d_model(img_size, num_classes, start_size=config['start_size'],
                                  kernel_size=config['kernel_size'], strides=config['strides'],
                                  intermediate_activation=config['intermediate_activation'])

    return model


def get_allopezr_3d(config, img_size, num_classes):
    img_size = (1,) + img_size
    model = get_allopezr_3d_model(img_size, num_classes, start_size=config['start_size'],
                                  kernel_size=config['kernel_size'], strides=config['strides'],
                                  intermediate_activation=config['intermediate_activation'])

    return model


def get_aspn_net(config, img_size, num_classes):
    model = get_aspn_model(img_size, num_classes, type='spn')

    return model


def get_fsk_net(config, img_size, num_classes):
    input_shape = img_size + (1,)
    model = get_fsk_model(input_shape, num_classes, start_size=config['start_size'])

    return model


def get_hybrid_sn(config, img_size, num_classes):
    input_shape = img_size + (1,)
    config_dict = training_config['hybrid_sn']

    return get_hybrid_sn_model(input_shape, num_classes, start_size=config_dict['start_size'])


def get_jigsaw_hsi(config, img_size, num_classes):
    input_shape = img_size + (1,)
    model = build_jigsaw_hsi(input_shape, num_classes=num_classes, crop=True, start_size=config['start_size'],
                             kernel_size=config['kernel_size'])

    return model


def get_lt_cnn(config, img_size, num_classes):
    model = get_lt_cnn_model(img_size, num_classes, start_size=config['start_size'], kernel_size=kernel_size,
                             strides=config['strides'])

    return model


def get_nezami_2020(config, img_size, num_classes):
    input_shape = img_size + (1,)

    return get_nezami_model(input_shape, num_classes, start_size=config['start_size'],
                            intermediate_activation=config['intermediate_activation'],
                            kernel_size=config['kernel_size'], strides=config['strides'])


def get_spectral_net(config, img_size, num_classes):
    model = get_wavelet_cnn_model(img_size, num_classes, start_size=config['start_size'],
                                  kernel_size=config['kernel_size'], pool_size=config['strides'])

    return model


dict_model = {
    'allopezr_2d': get_allopezr_2d,
    'allopezr_3d': get_allopezr_3d,
    'aspn': get_aspn_net,
    'fsk_net': get_fsk_net,
    'hybrid_sn': get_hybrid_sn,
    'jigsaw_hsi': get_jigsaw_hsi,
    'lt_cnn': get_lt_cnn,
    'nezami_2020': get_nezami_2020,
    'spectral_net': get_spectral_net,
}

hypertuner = {
    'allopezr_2d': get_kt_allopezr_2d_model
}

# ------------ TRAINING & COMPILATION -------------


def build_network(network_type, num_classes, image_dim):
    if network_type not in dict_model:
        raise ValueError('Unknown network type: {}'.format(network_type))

    print(training_config[network_type])

    return dict_model[network_type](training_config[network_type], image_dim, num_classes)


def compile_network(model, network_type, model_name, num_classes, show_summary=True, render_image=False):
    """
    Compiles the model and shows summary if required.
    """
    model.compile(optimizer=training_config[network_type]['optimizer'], loss=loss, metrics=get_metrics(num_classes))

    if show_summary:
        model.summary()

    if render_image:
        tf.keras.utils.plot_model(model, to_file=paths.result_folder + model_name + '.png', show_shapes=True,
                                  show_layer_names=True)

    return model


def run_model(model, X_train, y_train, callbacks, validation_split=0.1):
    """
    Fits the model with X_train and y_train.
    """

    print('Training for {} epochs with batch size of {}...'.format(epochs, batch_size))

    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, callbacks=callbacks)


def hypertune(X_train, y_train, network_type, model_name, callbacks, validation_split=0.1):
    """
    Performs hyperparameter tuning with the given hyperparameters.
    """
    if not network_type in hypertuner:
        print('No hypertuner for network type: {}'.format(network_type))
        return

    # Create a tuner
    tuner = kt.Hyperband(
        hypertuner[network_type],
        objective='val_loss',
        max_epochs=epochs,
        directory=paths.result_folder + 'tuning/' + model_name,
        project_name=model_name,
        seed=randomness.random_seed,
    )

    # Perform the search
    tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                 callbacks=callbacks)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters()[0]

    # Show the best hyperparameters
    print('Best hyperparameters:')
    print(best_hps.values)

    # Save into file
    with open(paths.result_folder + 'tuning/' + model_name + '/best_hps.txt', 'w') as f:
        f.write(str(best_hps.values))

    # Build the model with the best hyperparameters and train it on the data for 50 epochs
    # model = tuner.hypermodel.build(best_hps)
    # model.compile(optimizer=training_config[network_type]['optimizer'], loss=loss, metrics=get_metrics(num_classes))
    #
    # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
    #           callbacks=callbacks)


# -------------- System files ----------------
def read_json_config(path, network_type):
    with open(path, 'r') as f:
        data_params = json.load(f)
        if network_type not in data_params:
            raise ValueError('Unknown network type: {}'.format(network_type))

        network_config = data_params[network_type]
        if 'start_size' in network_config:
            training_config[network_type]['start_size'] = int(network_config['start_size'])
        if 'kernel_size' in network_config:
            training_config[network_type]['kernel_size'] = int(network_config['kernel_size'])
        if 'strides' in network_config:
            training_config[network_type]['strides'] = int(network_config['strides'])
        if 'intermediate_activation' in network_config:
            training_config[network_type]['intermediate_activation'] = network_config['intermediate_activation']
        if 'learning_rate' in network_config and 'optimizer' in training_config[network_type]:
            training_config[network_type]['optimizer'].learning_rate = float(network_config['learning_rate'])

        if 'batch_size' in data_params:
            cfg.batch_size = int(data_params['batch_size'])
        if 'epochs' in data_params:
            cfg.epochs = int(data_params['epochs'])
        if 'loss' in data_params:
            cfg.loss = data_params['loss']
        if 'last_activation' in data_params:
            cfg.last_activation = data_params['last_activation']
        if 'patch_size' in data_params:
            cfg.patch_size = int(data_params['patch_size'])
        if 'patch_overlapping' in data_params:
            cfg.patch_overlapping = int(data_params['patch_overlapping'])
        if 'test_split' in data_params:
            cfg.test_split = float(data_params['test_split'])
        if 'validation_split' in data_params:
            cfg.validation_split = float(data_params['validation_split'])
