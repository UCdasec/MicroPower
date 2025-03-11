# ASCAD_train_models.py - Modified 6/22/24 - Logan Reichling
# ASCAD_train_models.py original source available at https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_train_models.py
# Cite: L. Masure and R. Strullu, ‘Side Channel Analysis against the ANSSI’s protected AES implementation on ARM’. 2021.

import os
import os.path
import sys
import h5py
import numpy as np
import pandas as pd
import re
# Tensorflow Specific (+ turn off their logging)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, Conv2D, MaxPooling1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, AveragePooling1D, BatchNormalization, Activation, Add, add, AveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
tf.get_logger().setLevel('ERROR')


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def resnet_v1_2d(input_shape, depth, num_classes=256, ratio=1.0, without_permind=0):
    if (depth - 1) % 18 != 0:
        raise ValueError('depth should be 18n+1 (eg 19, 37, 55 ...)')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 1) / 18)
    inputs = Input(shape=input_shape)
    x = resnet_layer_2D(inputs=inputs, ratio=ratio)
    # Instantiate the stack of residual units
    for stack in range(9):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer_2D(inputs=x, num_filters=num_filters, strides=strides, ratio=ratio)
            y = resnet_layer_2D(inputs=y, num_filters=num_filters, activation=None, ratio=ratio)
            if stack > 0 and res_block == 0:
                x = resnet_layer_2D(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None,
                                 batch_normalization=False, ratio=ratio)
            x = add([x, y])
            x = Activation('relu')(x)
        if num_filters < 256:
            num_filters *= 2
    x = AveragePooling2D(pool_size=(4, 1))(x)
    x = Flatten()(x)
    x_alpha = alpha_branch(x, ratio=ratio)
    x_beta = beta_branch(x, ratio=ratio)
    x_sbox_l = []
    x_permind_l = []
    for i in range(16):
        x_sbox_l.append(sbox_branch(x, i, ratio=ratio))
        x_permind_l.append(permind_branch(x, i, ratio=ratio))
    if without_permind != 1:
        model = Model(inputs, [x_alpha, x_beta] + x_sbox_l + x_permind_l, name='extract_resnet')
    else:
        model = Model(inputs, [x_alpha, x_beta] + x_sbox_l, name='extract_resnet_without_permind')
    optimizer = Adam()  # Default LR is 0.001
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# 2D Layer for FPGAs
def resnet_layer_2D(inputs, num_filters=16, kernel_size=11, strides=1, activation='relu', batch_normalization=True,
                 conv_first=True, ratio=1.0):
    conv = Conv2D(int((num_filters * ratio) + 0.5), kernel_size=(kernel_size,1), strides=(strides),
                  padding='same', kernel_initializer='he_normal')
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


# Resnet layer sub-function of ResNetSCA
def resnet_layer(inputs, num_filters, kernel_size=11, strides=1, activation='relu', batch_normalization=True,
                 conv_first=True, ratio=1.0):
    x = inputs
    conv = Conv1D(int((num_filters * ratio) + 0.5), kernel_size=kernel_size, strides=strides, padding='same',
                  kernel_initializer='he_normal')
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


# Branch of ResNetSCA that predict the multiplicative mask alpha
def alpha_branch(x, ratio=1.0):
    x = Dense(int((1024 * ratio) + 0.5), activation='relu', name='fc1_alpha')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="softmax", name='alpha_output')(x)
    return x


# Branch of ResNetSCA that predict the additive mask beta
def beta_branch(x, ratio=1.0):
    x = Dense(int((1024 * ratio) + 0.5), activation='relu', name='fc1_beta')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="softmax", name='beta_output')(x)
    return x


# Branch of ResNetSCA that predict the masked sbox output
def sbox_branch(x, i, ratio=1.0):
    x = Dense(int((1024 * ratio) + 0.5), activation='relu', name='fc1_sbox_' + str(i))(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="softmax", name='sbox_' + str(i) + '_output')(x)
    return x


# Branch of ResNetSCA that predict the permutation indices
def permind_branch(x, i, ratio=1.0):
    x = Dense(int((1024 * ratio) + 0.5), activation='relu', name='fc1_pemind_' + str(i))(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation="softmax", name='permind_' + str(i) + '_output')(x)
    return x


# Generic function that produce the ResNetSCA architecture. Modified to include
# If without_permind option is set to 1, the ResNetSCA model is built without permindices branch
def resnet_v1(input_shape, depth, num_classes=256, ratio=1.0, without_permind=0):
    if (depth - 1) % 18 != 0:
        raise ValueError('depth should be 18n+1 (eg 19, 37, 55 ...)')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 1) / 18)
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=num_filters, ratio=ratio)
    # Instantiate the stack of residual units
    for stack in range(9):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides, ratio=ratio)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None, ratio=ratio)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None,
                                 batch_normalization=False, ratio=ratio)
            x = add([x, y])
            x = Activation('relu')(x)
        if num_filters < 256:
            num_filters *= 2
    x = AveragePooling1D(pool_size=4)(x)
    x = Flatten()(x)
    x_alpha = alpha_branch(x, ratio=ratio)
    x_beta = beta_branch(x, ratio=ratio)
    x_sbox_l = []
    x_permind_l = []
    for i in range(16):
        x_sbox_l.append(sbox_branch(x, i, ratio=ratio))
        x_permind_l.append(permind_branch(x, i, ratio=ratio))
    if without_permind != 1:
        model = Model(inputs, [x_alpha, x_beta] + x_sbox_l + x_permind_l, name='extract_resnet')
    else:
        model = Model(inputs, [x_alpha, x_beta] + x_sbox_l, name='extract_resnet_without_permind')
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def load_sca_model(model_file):
    check_file_exists(model_file)
    try:
        model = load_model(model_file)
    except:
        print("Error: can't load Keras model file '%s'" % model_file)
        sys.exit(-1)
    return model


# ASCAD helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the ASCAD
# database
def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)  # Load profiling traces
    Y_profiling = np.array(in_file['Profiling_traces/labels'])  # Load profiling labels
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)  # Load attacking traces
    Y_attack = np.array(in_file['Attack_traces/labels'])  # Load attacking labels
    if not load_metadata:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (
            in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])


def multilabel_to_categorical(Y):
    y = {}
    y['alpha_output'] = to_categorical(Y['alpha_mask'], num_classes=256)
    y['beta_output'] = to_categorical(Y['beta_mask'], num_classes=256)
    for i in range(16):
        y['sbox_' + str(i) + '_output'] = to_categorical(Y['sbox_masked'][:, i], num_classes=256)
    for i in range(16):
        y['permind_' + str(i) + '_output'] = to_categorical(Y['perm_index'][:, i], num_classes=16)
    return y


def multilabel_without_permind_to_categorical(Y):
    y = {}
    y['alpha_output'] = to_categorical(Y['alpha_mask'], num_classes=256)
    y['beta_output'] = to_categorical(Y['beta_mask'], num_classes=256)
    for i in range(16):
        y['sbox_' + str(i) + '_output'] = to_categorical(Y['sbox_masked_with_perm'][:, i], num_classes=256)
    return y


# Function for the finetuning
def finetuneModel(X_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100, multilabel=0,
                  validation_split=0, early_stopping=0):
    # Replace early stop with model checkpointer. Saves best val_loss like early stopping but more verbose
    check_file_exists(os.path.dirname(save_file_name))
    save_model = ModelCheckpoint(save_file_name, monitor='val_loss', verbose=True, save_best_only=True, mode='min')
    callbacks = [save_model]

    # Get the input layer shape
    if isinstance(model.get_layer(index=0).input_shape, list):
        input_layer_shape = model.get_layer(index=0).input_shape[0]
    else:
        input_layer_shape = model.get_layer(index=0).input_shape
    # Sanity check
    if input_layer_shape[1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (
            input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        Reshaped_X_profiling = X_profiling
    elif len(input_layer_shape) == 3:
        # This is a CNN: expand the dimensions
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)
    if multilabel == 1:
        y = multilabel_to_categorical(Y_profiling)
    elif multilabel == 2:
        y = multilabel_without_permind_to_categorical(Y_profiling)
    else:
        y = to_categorical(Y_profiling, num_classes=256)
    history = model.fit(x=Reshaped_X_profiling, y=y, batch_size=batch_size, verbose=1,
                        validation_split=0.1, epochs=epochs, callbacks=callbacks)
    return history


#### Training high level function
def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100, multilabel=0,
                validation_split=0, early_stopping=0):
    check_file_exists(os.path.dirname(save_file_name))
    # Save model calllback
    save_model = ModelCheckpoint(save_file_name)
    callbacks = [save_model]
    # Early stopping callback
    if (early_stopping != 0):
        if validation_split == 0:
            validation_split = 0.1
        callbacks.append(EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True))
    # Get the input layer shape
    if isinstance(model.get_layer(index=0).input_shape, list):
        input_layer_shape = model.get_layer(index=0).input_shape[0]
    else:
        input_layer_shape = model.get_layer(index=0).input_shape
    # Sanity check
    if input_layer_shape[1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (
            input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        Reshaped_X_profiling = X_profiling
    elif len(input_layer_shape) == 3:
        # This is a CNN: expand the dimensions
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)
    if multilabel == 1:
        y = multilabel_to_categorical(Y_profiling)
    elif multilabel == 2:
        y = multilabel_without_permind_to_categorical(Y_profiling)
    else:
        y = to_categorical(Y_profiling, num_classes=256)
    history = model.fit(x=Reshaped_X_profiling, y=y, batch_size=batch_size, verbose=1,
                        validation_split=validation_split, epochs=epochs, callbacks=callbacks)
    return history


def read_parameters_from_file(param_filename):
    # read parameters for the train_model and load_ascad functions
    param_file = open(param_filename, "r")

    my_parameters = eval(param_file.read())

    ascad_database = my_parameters["ascad_database"]
    training_model = my_parameters["training_model"]
    network_type = my_parameters["network_type"]
    epochs = my_parameters["epochs"]
    batch_size = my_parameters["batch_size"]
    train_len = 0
    if "train_len" in my_parameters:
        train_len = my_parameters["train_len"]
    validation_split = 0
    if "validation_split" in my_parameters:
        validation_split = my_parameters["validation_split"]
    multilabel = 0
    if "multilabel" in my_parameters:
        multilabel = my_parameters["multilabel"]
    early_stopping = 0
    if "early_stopping" in my_parameters:
        early_stopping = my_parameters["early_stopping"]

    pruning_ratio = None
    if "pruning_ratio" in my_parameters:
        pruning_ratio = my_parameters["pruning_ratio"]
    existing_model = None
    if "existing_model" in my_parameters:
        existing_model = my_parameters["existing_model"]
    ranks_path = None
    if "ranks_path" in my_parameters:
        ranks_path = my_parameters["ranks_path"]

    return (ascad_database, training_model, network_type, epochs, batch_size, train_len,
            validation_split, multilabel, early_stopping, pruning_ratio, existing_model, ranks_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # Force user to enter the file
        print("Utilize formatted file to train models")
        exit(1)
    else:
        # get parameters from user input
        (ascad_database, training_model, network_type, epochs, batch_size, train_len, validation_split, multilabel,
         early_stopping, pruning_ratio, existing_model, ranks_path) = read_parameters_from_file(sys.argv[1])

    # load traces
    (X_profiling, Y_profiling), (X_attack, Y_attack) = load_ascad(ascad_database)

    # get network type
    if network_type == "mlp":
        best_model = mlp_best(input_dim=len(X_profiling[0]))
    elif network_type == "cnn":
        best_model = cnn_best(input_dim=len(X_profiling[0]))
    elif network_type == "cnn2":
        best_model = cnn_best2(input_dim=len(X_profiling[0]))
    elif network_type == "multi_test":
        best_model = multi_test(input_dim=len(X_profiling[0]))
    elif network_type == "multi_resnet":
        best_model = resnet_v1((15000, 1), 19)
    elif network_type == "multi_resnet_without_permind":
        best_model = resnet_v1((15000, 1), 19, without_permind=1)
    elif network_type == "multi_resnet_prune_test":
        # Ensure existing_model and ranks_path are provided
        if existing_model is None or ranks_path is None or pruning_ratio is None:
            print("Error: Pruning_ratio, existing_model, and ranks_path must be provided for pruning test")
            sys.exit(1)
        # Copy weights from existing model into the pruned empty copy
        print((1 - float(pruning_ratio)))
        blank_pruned_best_model = resnet_v1((15000, 1), 19, without_permind=1, ratio=(1 - float(pruning_ratio)))
        # blank_pruned_best_model.summary()
        preTrainedModel = load_sca_model(existing_model)
        preTrainedModel.summary()
        best_model = copy_weights(preTrainedModel, blank_pruned_best_model, ranks_path)

    else:  # display an error and abort
        print("Error: no topology found for network '%s' ..." % network_type)
        sys.exit(-1)
    best_model.summary()

    # training
    if (train_len == 0):
        history = finetuneModel(X_profiling, Y_profiling, best_model, training_model, epochs, batch_size, multilabel,
                                validation_split, early_stopping)
    else:
        history = finetuneModel(X_profiling[:train_len], Y_profiling[:train_len], best_model, training_model, epochs,
                                batch_size, multilabel, validation_split, early_stopping)

    valAccLog = list()
    valAccLog.append(f'Alpha Val: {history.history["val_alpha_output_accuracy"][-1]}')
    valAccLog.append(f'Beta Val: {history.history["val_beta_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 0 Val: {history.history["val_sbox_0_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 1 Val: {history.history["val_sbox_1_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 2 Val: {history.history["val_sbox_2_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 3 Val: {history.history["val_sbox_3_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 4 Val: {history.history["val_sbox_4_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 5 Val: {history.history["val_sbox_5_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 6 Val: {history.history["val_sbox_6_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 7 Val: {history.history["val_sbox_7_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 8 Val: {history.history["val_sbox_8_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 9 Val: {history.history["val_sbox_9_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 10 Val: {history.history["val_sbox_10_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 11 Val: {history.history["val_sbox_11_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 12 Val: {history.history["val_sbox_12_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 13 Val: {history.history["val_sbox_13_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 14 Val: {history.history["val_sbox_14_output_accuracy"][-1]}')
    valAccLog.append(f'Sbox 15 Val: {history.history["val_sbox_15_output_accuracy"][-1]}')
    sboxValAccList = list()
    for i in range(16):
        sboxValAccList.append(history.history[f'val_sbox_{i}_output_accuracy'][-1])
    valAccLog.append(f'Sbox Val Min: {min(sboxValAccList)}')
    valAccLog.append(f'Sbox Val Avg: {sum(sboxValAccList) / len(sboxValAccList)}')

    import datetime

    with open(f"valAccLog{datetime.datetime.now()}", 'w') as f:
        for line in valAccLog:
            f.write(f"{line}\n")
