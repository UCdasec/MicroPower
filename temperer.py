# Imports
import argparse
import os
import sys
import time
from enum import Enum
import h5py
import numpy as np
import hashlib
import datetime
from tools import ASCADv2Adapter, process_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.platform import build_info
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

tf.get_logger().setLevel('ERROR')
import tools.loadData as lData


class LeakageModel(Enum):
    """
    Enum to represent the different leakage models we have available
    """

    def getEmbeddingSize(self, leakageModel):
        if leakageModel == self.HW:
            return 9
        elif leakageModel == self.ID:
            return 256

    # Enum values
    HW = 'HW'
    ID = 'ID'


def temperModel(model, patienceEpochs, maxEpochs, xProfiling, yProfiling, modelSaveFile, datasetName, verbose):
    """
    Takes an existing model and attempts to train longer to hit greater validation accuracies
    :return: None
    """

    # Set up tempering early stop (which automatically saves model to modelSaveFile during training)
    checkpointer = ModelCheckpoint(modelSaveFile)
    stopper = EarlyStopping(monitor="val_loss", patience=patienceEpochs, restore_best_weights=True)
    callbacks = [checkpointer, stopper]

    # Train model until val_loss stops decreasing (or until given max)
    if datasetName != "ASCADv2":
        history = model.fit(x=xProfiling, y=yProfiling, validation_split=0.1, batch_size=100, verbose=verbose,
                            epochs=maxEpochs, shuffle=True, callbacks=callbacks)
    else:
        history = model.fit(x=xProfiling, y=yProfiling, validation_split=0.05, batch_size=64, verbose=verbose,
                            epochs=maxEpochs, shuffle=True, callbacks=callbacks)

    # Export additional statistics and reproducibility information
    # createLossGraph(self.outputDir, self.datasetName, "tempering_loss", 1, history.history["loss"],
    #                 history.history["val_loss"])
    # self.exportReproducibilityStats(timeDelta, history.history['loss'][-1], len(history.history["val_loss"]),
    #                                 modelSaveFile)
    print('[LOG] -- Done!')
    return history


def exportReproducibilityStats(datasetName, inputTracePath, timeDelta, maxTrainLoss, finalEpochs, modelOutputDir,
                               modelFilePath, trainTracesNum, attackWindow, targetByte, verbose):
    """
    Export reproducibility statistics to a file in the output directory after pruning+finetuning is complete
    :param datasetName: Name of the dataset which is used for the tempering
    :param inputTracePath: Path to the utilized traces
    :param timeDelta: Floating point time delta representing the time it took to complete training
    :param maxTrainLoss: Final loss values from the training
    :param finalEpochs: Final utilized epochs from the process
    :param modelOutputDir: Folder path containing trained model
    :param modelFilePath: File path to the trained model
    :param trainTracesNum: Number of traces used for training
    :param attackWindow: Attack window used for the traces
    :param targetByte: Target byte used for the traces
    :param verbose: Reported verbose mode (true or false)
    :return: None
    """
    # Ensure output directory for trained model was actually created
    if not os.path.exists(modelOutputDir):
        print('[FATAL] -- Output directory for trained model was not created for training.')
        exit(1)
    if not os.path.exists(modelFilePath):
        print('[FATAL] -- Output trained model file was not created.')
        exit(1)

    reprodOutputLogFile = os.path.join(modelOutputDir, f'pruneTune_{datasetName}_{datetime.date.today()}.log')
    reprodLog = list()
    reprodLog.append(f"{os.path.join(modelOutputDir, f'pruneTune_{datasetName}_{datetime.date.today()}.log')}")
    reprodLog.append(f"Pruning and finetuning completed on {time.ctime(time.time())}")
    reprodLog.append(f"Model saved to {modelOutputDir}")
    reprodLog.append(f"Pruning script run with the following command:")
    reprodLog.append("python3 " + " ".join(sys.argv))
    reprodLog.append(f"Process took {timeDelta:.2f} seconds with {finalEpochs} epochs")
    reprodLog.append(f"Final loss: {maxTrainLoss}")
    reprodLog.append(f" -------------- Current library versions: --------------")
    reprodLog.append(f"Python: {sys.version}")
    reprodLog.append(f"Tensorflow: {tf.__version__}")
    reprodLog.append(f"NVidia CUDA Runtime version: {build_info.build_info['cuda_version']}")
    reprodLog.append(f"NVidia CUDNN Runtime version: {build_info.build_info['cudnn_version']}")
    reprodLog.append(f" -------------- Current train object parameters: --------------")
    reprodLog.append(f"Model file export path: {modelFilePath}")
    reprodLog.append(f"Model hash (SHA256): {hashlib.sha256(open(modelFilePath, 'rb').read()).hexdigest()}")
    reprodLog.append(f"Dataset file: {inputTracePath}")
    reprodLog.append(f"Dataset name: {datasetName}")
    reprodLog.append(
        f"Dataset hash (SHA256): {hashlib.sha256(open(inputTracePath, 'rb').read()).hexdigest()}")
    reprodLog.append(f"Train traces: {trainTracesNum}")
    reprodLog.append(f"Attack window: {attackWindow}")
    reprodLog.append(f"Target byte: {targetByte}")
    reprodLog.append(f"Leakage model: {leakageModel.value}")
    reprodLog.append(f"Verbose: {verbose}")
    reprodLog.append(f"")

    # Write each line to the log file
    with open(reprodOutputLogFile, 'w') as f:
        for line in reprodLog:
            f.write(f"{line}\n")


def loadData(datasetFileExtension, datasetName, inputTracesFile, attackWindow, preprocess, trainTracesNum, targetByte,
             leakModel, fpgaMode=None, ascadV2Type=None):
    """
    Class function designed for use with an initialized SideChannelTrainer object.
    Load profiling and training data from the given npz traces file.
    :return: Traces (x values for model, power value sampled),
        labels (y values for model, intermediate AES representation),
        plaintext: original plaintext string,
        key: 128-bit key in hex string,
        inp_shape: Shape of x values traces to compare against model
    """
    if datasetFileExtension == 'npz':  # Our format
        whole_pack = np.load(inputTracesFile)
        traces, plaintext, key = lData.load_data_base(whole_pack, attackWindow, preprocess,
                                                      trainTracesNum, shifted=0)
        labels = lData.get_labels(plaintext, key[targetByte], targetByte, leakModel.value)
        labels = to_categorical(labels, leakModel.getEmbeddingSize(leakModel))
        inp_shape = (traces.shape[1], 1) # if not fpgaMode else (traces.shape[1], 1, 1)
    elif datasetFileExtension == 'h5' and datasetName == 'ASCAD':  # ASCAD format:
        print("[WARNING] -- Loaded ASCAD database ignores attack window, target byte, preprocess and shifted")
        attackWindow = "0_700"
        targetByte = 2
        preprocess = ""
        with h5py.File(inputTracesFile) as in_file:
            traces = np.array(in_file['Profiling_traces/traces'])[:trainTracesNum, :]
            labels = np.array(in_file['Profiling_traces/labels'], dtype='uint8')[:trainTracesNum]
            plaintext = np.array(in_file['Profiling_traces/metadata'])['plaintext'][:trainTracesNum]
            key = np.array(in_file['Profiling_traces/metadata'])['key'][0]
            labels = to_categorical(labels, leakModel.getEmbeddingSize(leakModel))
            inp_shape = (traces.shape[1], 1)
    elif datasetFileExtension == 'h5' and datasetName == 'CHES':  # CHES format:
        print("[Notice] -- Loaded CHES database for training")
        with h5py.File(inputTracesFile) as in_file:
            # Set attack window:
            tmp = attackWindow.split('_')
            attackWindow = [int(tmp[0]), int(tmp[1])]
            traces = np.array(in_file['Profiling_traces/traces'])[:trainTracesNum, attackWindow[0]:attackWindow[1]]
            plaintext = np.array(in_file['Profiling_traces/metadata'])['plaintext'][:trainTracesNum]
            plaintext = plaintext.astype(np.uint8)
            key = np.array(in_file['Profiling_traces/metadata'])['key'][0]  # Just get first key since its fixed
            key = key.astype(np.uint8)
            labels = lData.get_labels(plaintext, key[targetByte], targetByte, leakModel.value)
            labels = to_categorical(labels, leakModel.getEmbeddingSize(leakModel))
            inp_shape = (traces.shape[1], 1)
    elif datasetFileExtension == "h5" and datasetName == "ASCADv2":
        # TracesX[0]: train, TracesX[1]: eval | TracesY[0]: train, TracesY[1]: eval | inp_shape
        traces, labels, inp_shape = ASCADv2Adapter.loadData(inputTracesFile, trainTracesNum,0, ascadV2Type)
        traces = traces[0]  # Training dataset slice only (no validation cutoff for iterative pruning)
        labels = labels[0]  # Training dataset slice only (no validation cutoff for iterative pruning)

    return traces, labels, inp_shape


def parseArgs():
    """
    Parse command line arguments if run from command line
    :return: Namespace object containing parsed arguments (i.e. args = parser.parse_args(); args.input;)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_traces', help='Input traces to train the model')
    parser.add_argument('-m', '--input_model_dir', help='Input model directory to train further')
    parser.add_argument('-o', '--output_dir', help='Output directory for the trained model')
    parser.add_argument('-tn', '--train_traces', type=int, help='Number of traces to train the model')
    parser.add_argument('-tb', '--target_byte', type=int, help='Target byte to attack (0-15)')
    parser.add_argument('-me', '--max_epochs', type=int, help='Max epochs to run early stopper')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'ID'}, help='Leakage model of the network')
    parser.add_argument('-aw', '--attack_window', help='Attack window (POI window) for the traces')
    parser.add_argument('-ascadV2Type', '--ascadV2Type', choices={"withPermIDs", "withoutPermIDs"},
                        help='Selects the type of ASCADv2 ResNet to generate')
    parser.add_argument('-fpga', '--fpga', action='store_true', help='For use with 2D conv layer version')
    parser.add_argument('-v', '--verbose', action='store_true', help='Include for verbose output')
    parserOptions = parser.parse_args()
    return parserOptions


if __name__ == '__main__':
    opts = parseArgs()

    loadedModel = load_model(os.path.join(opts.input_model_dir, 'model', 'best_model.h5'))
    loadedModel.summary()

    # Create output directory and output file path
    modelDir = os.path.join(opts.output_dir, 'model')
    os.makedirs(modelDir, exist_ok=True)
    modelSaveFile = os.path.join(modelDir, 'best_model.h5')

    if opts.leakage_model == "HW":
        leakageModel = LeakageModel.HW
    else:
        leakageModel = LeakageModel.ID

    _, fileName = os.path.split(opts.input_traces)
    datasetName = fileName.split('.')[0]
    datasetFileExtension = fileName.split('.')[1]

    X_profiling, Y_profiling, input_shape = loadData(datasetFileExtension, datasetName, opts.input_traces,
                                                     opts.attack_window,'', opts.train_traces,
                                                     opts.target_byte, leakageModel, opts.fpga, opts.ascadV2Type)

    input_layer_shape = loadedModel.get_layer(index=0).input_shape
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]
    print(input_layer_shape)
    Reshaped_X_profiling = process_data.sanity_check(input_layer_shape, X_profiling)

    # Start training timer
    startTemperTime = time.time()
    history = temperModel(loadedModel, opts.max_epochs/10, opts.max_epochs, Reshaped_X_profiling, Y_profiling,
                          modelSaveFile, datasetName, opts.verbose)
    endTemperTime = time.time()
    timeDelta = endTemperTime - startTemperTime
    print(f'Tempered for {len(history.history["val_loss"])} epochs, {timeDelta:.4f} seconds')
    print(f"Time per epoch: {timeDelta / len(history.history['val_loss']):.4f} seconds")

    print("Exporting reproducibility stats...")
    exportReproducibilityStats(datasetName, opts.input_traces, timeDelta, history.history['loss'],
                               len(history.history["val_loss"]), opts.output_dir, modelSaveFile, opts.train_traces,
                               opts.attack_window, opts.target_byte, opts.verbose)
