# model_zoo.py - Start 12/13/23 - UC DASEC - Logan Reichling
# Contains model architectures under test, including the ASCADv1 and ASCADv2 neural networks wrapped in our framework

import os
import numpy as np
import pandas as pd
# Tensorflow Specific (+ turn off their logging)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Dense, Conv1D, Input, AveragePooling1D, Flatten, BatchNormalization, \
    GlobalAveragePooling1D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, AveragePooling1D, BatchNormalization, Activation, Add, add, Conv2D, AveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# CNN Best model from ASCADv1 (with framework pruning modifications added)
# Cite: E. Prouff, R. Strullu, R. Benadjila, E. Cagli, and C. Dumas, ‘Study of Deep Learning Techniques for
# Side-Channel Analysis and Introduction to ASCAD Database’. 2018.
def cnn_best(input_shape, ratio=[1] * 7, initialSize=None, emb_size=9, classification=True):
    # r = [1] * 7
    # print(ratio)
    inp = Input(shape=input_shape)
    # 'Normal' rounding on each, then if filter size would be below 1, set to 1
    # Block 1
    x = Conv1D(int(((64 if initialSize is None else initialSize[0]) * ratio[0]) + 0.5)
               if (int(((64 if initialSize is None else initialSize[0]) * ratio[0]) + 0.5)) >= 1 else 1,
               11, strides=2, activation='relu', padding='same', name='block1_conv1')(inp)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # x = BatchNormalization()(x)
    # Block 2
    x = Conv1D(int(((128 if initialSize is None else initialSize[1]) * ratio[1]) + 0.5)
               if (int(((128 if initialSize is None else initialSize[1]) * ratio[1]) + 0.5)) >= 1 else 1
               , 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # x = BatchNormalization()(x)
    # Block 3
    x = Conv1D(int(((256 if initialSize is None else initialSize[2]) * ratio[2]) + 0.5)
               if (int(((256 if initialSize is None else initialSize[2]) * ratio[2]) + 0.5)) >= 1 else 1
               , 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # x = BatchNormalization()(x)
    # Block 4
    x = Conv1D(int(((512 if initialSize is None else initialSize[3]) * ratio[3]) + 0.5)
               , 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # x = BatchNormalization()(x)
    # Block 5
    x = Conv1D(int(((512 if initialSize is None else initialSize[4]) * ratio[4]) + 0.5)
               , 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # x = BatchNormalization()(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(int(((4096 if initialSize is None else initialSize[5]) * ratio[5]) + 0.5)
              , activation='relu', name='fc1')(x)
    x = Dense(int(((4096 if initialSize is None else initialSize[6]) * ratio[6]) + 0.5)
              , activation='relu', name='fc2')(x)
    if classification:
        x = Dense(emb_size, activation='softmax', name='predictions')(x)
        model = Model(inp, x, name='cnn_best')
        optimizer = RMSprop(learning_rate=0.00001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    else:
        x = Dense(emb_size, kernel_initializer="he_normal")(x)
        model = Model(inp, x, name='cnn_best')
        return model


def cnn_best_fpga2D(input_shape, ratio=[1] * 7, initialSize=None, emb_size=256, classification=True):
    inp = Input(shape=(*input_shape, 1))

    # Block 1
    x = Conv2D(int(((64 if initialSize is None else initialSize[0]) * ratio[0]) + 0.5)
               if (int(((64 if initialSize is None else initialSize[0]) * ratio[0]) + 0.5)) >= 1 else 1,
               (11, 1), strides=(2, 1), activation='relu', padding='same', name='block1_conv2')(inp)
    x = AveragePooling2D((2, 1), strides=(2, 1), name='block1_pool')(x)
    # x = BatchNormalization()(x)
    # Block 2
    x = Conv2D(int(((128 if initialSize is None else initialSize[1]) * ratio[1]) + 0.5)
               if (int(((128 if initialSize is None else initialSize[1]) * ratio[1]) + 0.5)) >= 1 else 1,
               (11, 1), activation='relu', padding='same', name='block2_conv2')(x)
    x = AveragePooling2D((2, 1), strides=(2, 1), name='block2_pool')(x)
    # x = BatchNormalization()(x)
    # Block 3
    x = Conv2D(int(((256 if initialSize is None else initialSize[2]) * ratio[2]) + 0.5)
               if (int(((256 if initialSize is None else initialSize[2]) * ratio[2]) + 0.5)) >= 1 else 1,
               (11, 1), activation='relu', padding='same', name='block3_conv2')(x)
    x = AveragePooling2D((2, 1), strides=(2, 1), name='block3_pool')(x)
    # x = BatchNormalization()(x)
    # Block 4
    x = Conv2D(int(((512 if initialSize is None else initialSize[3]) * ratio[3]) + 0.5),
               (11, 1), activation='relu', padding='same', name='block4_conv2')(x)
    x = AveragePooling2D((2, 1), strides=(2, 1), name='block4_pool')(x)
    # x = BatchNormalization()(x)
    # Block 5
    x = Conv2D(int(((512 if initialSize is None else initialSize[3]) * ratio[3]) + 0.5),
               (11, 1), activation='relu', padding='same', name='block5_conv2')(x)
    x = AveragePooling2D((2, 1), strides=(2, 1), name='block5_pool')(x)
    # x = BatchNormalization()(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(int(((4096 if initialSize is None else initialSize[6]) * ratio[6]) + 0.5),
              activation='relu', name='fc1')(x)
    x = Dense(int(((4096 if initialSize is None else initialSize[6]) * ratio[6]) + 0.5),
              activation='relu', name='fc2')(x)
    if classification:
        x = Dense(emb_size, activation='softmax', name='predictions')(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        optimizer = RMSprop(learning_rate=0.00001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print('[log] --- finish construct the cnn2 model')
        return model
    else:
        # embeddings = x
        x = Dense(emb_size, kernel_initializer="he_normal")(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        return model


def copy_weights(pre_trained_model, target_model, ranks_path):  # unused in favor of new copy function
    ranks = pd.read_csv(ranks_path, header=None).values
    rr = list()
    for r in ranks:
        r = r[~np.isnan(r)]
        r = list(map(int, r))
        rr.append(r)
    for l_idx, l in enumerate(target_model.layers):
        if "conv" in l.name:
            conv_id = int(l.name[5]) - 1
            this_idcies = rr[conv_id][1:l.filters + 1]
            if conv_id == 0:
                # print(f"Num layers pre-trained model: {len(pre_trained_model.layers)}")
                # print(f"Num layers target model: {len(target_model.layers)}")
                weights = pre_trained_model.layers[l_idx].get_weights()[0][:, :, this_idcies]
            else:
                last_idcies = rr[conv_id - 1][1:last_filters + 1]
                weights = pre_trained_model.layers[l_idx].get_weights()[0][:, :, this_idcies]
                weights = weights[:, last_idcies, :]
            bias = pre_trained_model.layers[l_idx].get_weights()[1][this_idcies]
            l.set_weights([weights, bias])
            last_filters = l.filters
    return target_model


def applyPTQ(preTrainedModel, quantizationType: str, poiWidthSize=None, trainingTraces=None):
    """
    ApplyPTQ - Apply Post Training Quantization to a model using Tensorflow's TFLiteConverter
    :param preTrainedModel: The model to be quantized (and converted to TFLite)
    :param quantizationType: The type of quantization to be applied (tfliteOnly, dynamic, dynamicWRep, float16, int8, uint8)
    :param poiWidthSize: Size the POI window in use of the representative dataset.
    :param trainingTraces: A representative dataset for dynamicWRep, int8, and uint8 quantization
    :return: A TFLite model with the specified quantization applied
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(preTrainedModel)

    # The code per quantization type is based on examples from Tensorflow's documentation
    if quantizationType == "tfliteOnly":
        pass

    elif quantizationType == "dynamicInt8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif quantizationType == "dynamicWRepInt8":
        if trainingTraces is None:
            print("Must provide representative dataset for dynamicWRep quantization")
            exit(1)
        def representative_dataset():
            for i in range(500):  # Just 500 based on TF's recommendation
                yield [trainingTraces[i].reshape(1, poiWidthSize, 1).astype(np.float32)]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset

    elif quantizationType == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quantizationType == "int8":
        if trainingTraces is None:
            print("Must provide representative dataset for int8 quantization")
            exit(1)
        def representative_dataset():
            for i in range(500):  # Just 500 based on TF's recommendation
                yield [trainingTraces[i].reshape(1, poiWidthSize, 1)]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    elif quantizationType.lower() == "uint8":
        if trainingTraces is None:
            print("Must provide representative dataset for uint8 quantization")
            exit(1)
        def representative_dataset():
            for i in range(500):  # Just 500 based on TF's recommendation
                yield [trainingTraces[i].reshape(1, poiWidthSize, 1)]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_quant_model = converter.convert()
    return tflite_quant_model
