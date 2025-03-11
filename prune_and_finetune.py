# prune_and_finetune.py - Start 12/13/23 - UC DASEC - Logan Reichling
# Applies various pruning and finetuning techniques on a model

# General imports
import argparse
import gc
import os
import re
import time
from datetime import date
import hashlib
import h5py
import sys
import matplotlib.pyplot as plt
import numpy as np

from tools import model_zoo
from tools import FPGM_L2_Score
from tools import genInfoPlots
from tools import architectureExtractor
from tools import ASCADv2Adapter
from tools import process_data
from tools.SideChannelConstants import SideChannelConstants, LeakageModel

# Tensorflow Specific (+ turn off their logging)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.platform import build_info
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print("Mem growth error", e)


def createLossGraph(exportDir, datasetName, saveName, rounds, lossValues, valLossValues):
    """
    Creates a graph of the val_loss and loss and saves it to the given export directory, for informational purposes
    :param exportDir: Directory in which to export the figures
    :param datasetName: Name of the dataset used in training
    :param saveName: File name for the generated graph
    :param rounds: Number of rounds in the iterative pruning
    :param lossValues: Loss values from history object from model.fit() -> history.history['loss']
    :param valLossValues: Loss values from history object from model.fit() -> history.history['val_loss']
    :return: None
    """
    plt.plot(lossValues)
    plt.plot(valLossValues)
    plt.title(f'Training Loss w/ {datasetName} Dataset over {rounds} round(s)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(exportDir, saveName + '.png'))
    plt.savefig(os.path.join(exportDir, saveName + '.pdf'))
    plt.show()
    plt.clf()


def getModelSize(model):
    """
    Given a H5 Keras model, return an array of model filter sizes for convolutional and dense layers
    :param model: Keras/Tensorflow model object loaded into memory
    :return: Array of integer filter sizes
    """
    layerShapes = list()
    for layer in model.layers:
        if "conv" in layer.name or "fc" in layer.name:
            layerShapes.append(layer.output_shape[-1])
    return layerShapes


def train_model(Reshaped_X_profiling, Y_profiling, trainingTracesIndicesPerRound, model, save_file_name, epochs,
                verbose=False, datasetType="CNN"):
    """
    High level function to train a model and save it to a file
    :param Reshaped_X_profiling: Reshaped (for the model's architecture) X profiling training traces
    :param Y_profiling: Labels for the profiling traces
    :param trainingTracesIndicesPerRound: Tuple of start and end indices for the training traces
    :param model: A Keras model to train
    :param save_file_name: Output model file save name
    :param epochs: Number of epochs to train for
    :param verbose: Optional verbosity flag
    :param datasetType: Type of dataset being used
    :return: History object from model.fit()
    """
    if datasetType != "ASCADv2":
        checkpointer = ModelCheckpoint(save_file_name, monitor='val_accuracy', verbose=verbose, save_best_only=True,
                                       mode='max')  # Save model with the best validation accuracy
        batch_size = 100  # Set batch size to common 100
        history = model.fit(x=Reshaped_X_profiling[trainingTracesIndicesPerRound[0]:trainingTracesIndicesPerRound[1]],
                            y=Y_profiling[trainingTracesIndicesPerRound[0]:trainingTracesIndicesPerRound[1]],
                            validation_split=0.1, batch_size=batch_size, verbose=verbose, epochs=epochs, shuffle=True,
                            callbacks=[checkpointer])
    else:  # datasetType == "ASCADv2"
        checkpointer = ModelCheckpoint(save_file_name, monitor='val_loss', verbose=verbose, save_best_only=True,
                                       mode='min')  # Save model with the best validation accuracy
        batch_size = 64  # ASCADv2 batch size
        history = model.fit(x=Reshaped_X_profiling[0], y=Y_profiling[0], validation_split=0.05, batch_size=batch_size,
                            verbose=verbose, epochs=epochs, shuffle=True, callbacks=[checkpointer])
    print('[LOG] -- Model save to path: {}'.format(save_file_name)) if verbose else None
    return history


def printRoundHeader(i, endCondition, iterativeType, roundRatios, epochsPerRound, modelSizes):
    """
    Print the header for a given pruning round
    :param i: Current round number
    :param endCondition: End round number for iterative pruning (or automatic)
    :param iterativeType: Type of iterative pruning {'manual', 'automatic'}
    :param roundRatios: Current pruning ratios for the round
    :param epochsPerRound: Current number of epochs to train for the subround
    :param modelSizes: Current model filter sizes
    :return: None
    """
    print(f"------- Round {i} of {endCondition if iterativeType != 'automatic' else '*Automatic*'} ------")
    print(f"Curr. pruning ratios: ", end="")
    for pruningRate in roundRatios:
        print(f"{pruningRate:.1f} ", end="")
    print(f"| Training {epochsPerRound} epochs ", end="")
    print(f"| Curr. filter sizes: ", end="")
    for filterSize in modelSizes:
        print(f"{filterSize} ", end="")
    print("")


def checkIfFilterSizeSame(currFilterSizes, roundRatios):
    """
    Check if the filter sizes are the same before and after pruning
    :param currFilterSizes: Current filter sizes
    :param roundRatios: Pruning ratios applied to the current filter sizes
    :return: Boolean value indicating if the filter sizes are the same
    """
    filterSizesAfter = [int(currFilterSizes[x] * ratio + 0.5) for x, ratio in enumerate(roundRatios)]
    filtersSame = 0
    for f in range(len(currFilterSizes)):
        if currFilterSizes[f] == filterSizesAfter[f]:
            filtersSame += 1
    if filtersSame == len(currFilterSizes):
        return True
    return False


def stoppingCriteria1(currFilterSizes, roundRatios, endCondition):
    """
    Stopping criterion 1: Filters unchanged after pruning ratio applied
    :param currFilterSizes: Current sizes of the model filters
    :param roundRatios: Ratios to be applied against the current filter sizes
    :param endCondition: Loop control variable
    :return:
    """
    if checkIfFilterSizeSame(currFilterSizes, roundRatios):
        print(f"Stopping criterion reached, filters unchanged after pruning ratio applied.")
        newEndCondition = endCondition - 1  # To let loop end naturally
    else:
        newEndCondition = endCondition
    return newEndCondition


def stoppingCriteria2(outputDir, i, endCondition):
    """
    Stopping Critiera 2: If lower than the minimum pruning rate is reached, stop the iterative pruning process
    :param outputDir: Output directory for the model
    :param i: Round number
    :param endCondition: Loop control variable
    :return: Directories to save the failed model as well as metadata
    """
    print(f"Stopping criterion reached, failure at minimum pruning rate.")
    currentRoundDir = os.path.join(outputDir, f'Round{i}')
    os.rename(currentRoundDir, currentRoundDir + "Failed")
    model_save_file = os.path.join(outputDir, f'Round{i}Failed', 'model', 'best_model.h5')
    newEndCondition = endCondition - 1
    return currentRoundDir, model_save_file, newEndCondition


class SideChannelPrunerAndFinetuner:
    """
    Class which prunes and finetunes a given NN side channel model using the passed parameters
    """

    def __init__(self, inputTracesFile: str, inputSeedModelDir: str, outputDir: str, epochs: str, targetByte: int,
                 leakageModel: str, architecture: str, attackWindow: str, pruningType: str, trainTracesNum: int,
                 valTracesNum: int, valAccuracy: float, filterScore: str, quantizationType: str, ranksPath: str,
                 ascadV2Type: str, fpga: bool = False, verbose: bool = False):
        self.inputTracesFile = inputTracesFile
        self.inputSeedModelDir = inputSeedModelDir
        self.model = None
        self.outputDir = outputDir
        self.epochs = epochs
        self.targetByte = targetByte
        self.leakageModel = leakageModel
        self.architecture = architecture
        self.attackWindow = attackWindow
        self.pruningType = pruningType
        self.trainTracesNum = trainTracesNum
        self.valTracesNum = valTracesNum
        self.valAccuracy = valAccuracy
        self.filterScore = filterScore
        self.quantizationType = quantizationType
        self.ranksPath = ranksPath
        self.ascadV2Type = ascadV2Type
        self.fpgaArchitecture = fpga
        self.verbose = verbose

        # Set later
        self.logs = None
        self.datasetName = None
        self.datasetFileExtension = None

        # Ensure that the required parameters are valid and set remaining internal variables
        self.verifyParameters()

        # Calculate additional parameters and final parameter checks
        if leakageModel == 'HW':
            self.leakageModel = LeakageModel.HW
        elif leakageModel == 'ID':
            self.leakageModel = LeakageModel.ID

        if os.path.exists(os.path.join(self.inputSeedModelDir, 'model', 'best_model.tflite')):
            print('[FATAL] -- TFLite models are not supported for additional training (at this point).')
            exit(1)
        elif os.path.exists(os.path.join(self.inputSeedModelDir, 'model', 'best_model.h5')):
            self.inputModelFile = os.path.join(self.inputSeedModelDir, 'model', 'best_model.h5')
            self.model = load_model(self.inputModelFile)

        # Generate FPGM or L2 ranks if needed
        if self.ranksPath == "generate":
            scoreOutputDir = self.outputDir
            if self.filterScore == "fpgm":
                FPGM_L2_Score.fpgm_scores(self.model, scoreOutputDir, False)
                self.ranksPath = os.path.join(scoreOutputDir, 'fpgm_idx.csv')
            else:
                FPGM_L2_Score.l2_scores(self.model, scoreOutputDir, False)
                self.ranksPath = os.path.join(scoreOutputDir, 'l2_idx.csv')
        # print(self.ranksPath)

    def verifyParameters(self):
        """
        Lengthy function to verify that the training parameters are valid and give helpful output if incorrect.
        EXIT if invalid.
        :return: None
        """
        exitFlag = False

        if self.inputTracesFile is None:
            print('[FATAL] -- Missing required input traces file parameter, exiting.')
            exitFlag = True
        elif not os.path.exists(self.inputTracesFile):
            print('[FATAL] -- Input traces file does not exist.')
            exit(1)

        _, fileName = os.path.split(self.inputTracesFile)  # Extract dataset name for metadata use (e.g. X1_K1_200k)
        self.datasetName = fileName.split('.')[0]
        self.datasetFileExtension = fileName.split('.')[1]
        if self.datasetFileExtension != 'npz' and self.datasetFileExtension != 'h5':
            print(f'[FATAL] -- Input traces file {self.datasetName} must be a .npz or .h5 file.')
            exit(1)
        datasetMatch = re.compile(SideChannelConstants.getDatasetNamingConvention())
        if datasetMatch.match(self.datasetName) is None and self.datasetName not in ['ASCAD', 'ASCADv2']:
            print(f'[FATAL] -- Input traces "{self.datasetName}" file unrecognized. Modify python file if needed.')
            exit(1)

        if self.inputSeedModelDir is None:
            print('[FATAL] -- Missing required input seed model directory parameter, exiting.')
            exit(1)
        elif not os.path.exists(self.inputSeedModelDir):
            print('[FATAL] -- Input seed model directory does not exist.')
            exitFlag = True

        if self.outputDir is None:
            print('[FATAL] -- Missing required output model directory parameter, exiting.')
            exit(1)
        if self.datasetName is not None and self.datasetName != 'ASCADv2':
            if self.targetByte is None:
                print('[FATAL] -- Missing required target byte parameter, exiting.')
                exit(1)
            if int(self.targetByte) < 0 or (self.targetByte > 15):
                print(f'[FATAL] -- Target byte {self.targetByte} must be between 0 and 15.')
                exitFlag = True

        if self.leakageModel is None:
            print('[FATAL] -- Missing required leakage model parameter, exiting.')
            exit(1)
        if self.leakageModel != "HW" and self.leakageModel != "ID":
            print(f'[FATAL] -- Leakage model {self.leakageModel} unrecognized. Must be ID or HW.')
            exitFlag = True
        if self.attackWindow is None:
            print('[FATAL] -- Missing required attack window parameter, exiting.')
            exit(1)
        if self.pruningType is None:
            print('[FATAL] -- Pruning type not specified, exiting.')
            exit(1)

        if self.pruningType == 'iterative':
            if self.trainTracesNum is None:
                print('[FATAL] -- Number of training traces not specified, exiting.')
                exit(1)
            if self.trainTracesNum < 0:
                print('[FATAL] -- Train trace number must be greater than 0.')
                exitFlag = True
            if self.ranksPath is None:
                print('[LOG] -- Ranks path not provided, generating...')
                self.ranksPath = "generate"
            elif not os.path.exists(self.ranksPath):
                print(f'[FATAL] -- Ranks file does not exist at {self.ranksPath}.')
                exitFlag = True
            if self.valTracesNum is None:
                print('[FATAL] -- Number of testing traces not specified for iterative pruning, exiting.')
                exit(1)
            if self.valTracesNum < 0:
                print('[FATAL] -- Validation trace number must be greater than 0.')
                exitFlag = True
            if self.valAccuracy is None:
                print('[FATAL] -- Testing accuracy not specified for iterative pruning, exiting.')
                exit(1)
            if self.valAccuracy < 0:
                print('[FATAL] -- Test accuracy must be greater than 0.')
                exitFlag = True
            if self.epochs is None:
                print('[FATAL] -- Missing required epochs parameter, exiting.')
                exit(1)
            try:
                _ = int(self.epochs)
            except ValueError:
                print(f'[FATAL] -- Epochs "{self.epochs}" in bad format for automatic iterative pruning.')
                exitFlag = True
        elif self.pruningType == 'ptq':
            if self.quantizationType is None:
                print('[FATAL] -- Quantization type not specified for PTQ, exiting.')
                exit(1)
        if exitFlag:
            exit(1)  # Exit if any of the above parameters are invalid

    def simpleLog(self, message, verbose=False):
        """
        Simple logging function to append a message to the logs list
        :param message: String message to append to the logs list
        :param verbose: Optional verbosity flag
        :return: None
        """
        if self.logs is None:
            self.logs = list()
        self.logs.append(message)
        if self.verbose or verbose:
            print(message)

    def exportReproducibilityStats(self, timeDelta, roundTimes, pruningRatesPerRound, ESSR, maxTrainLoss, lenLoss,
                                   finalEpochs, modelOutputDir, modelFilePath):
        """
        Export reproducibility statistics to a file in the output directory after pruning+finetuning is complete
        :param timeDelta: Floating point time delta representing the time it took to complete training
        :param roundTimes: Floating point list of times representing the time per round
        :param pruningRatesPerRound: Floating point list of pruning rates utilized in each round
        :param ESSR: Floating point value representing the Equivalent Single Shot Ratio
        :param maxTrainLoss: Final loss values from the training
        :param lenLoss: Length of the loss values from the training to get exact epochs
        :param finalEpochs: Final utilized epochs from the process
        :param modelOutputDir: Folder path containing trained model
        :param modelFilePath: File path to the trained model
        :return: None
        """
        # Ensure output directory for trained model was actually created
        if not os.path.exists(modelOutputDir):
            print('[FATAL] -- Output directory for trained model was not created for training.')
            exit(1)
        if not os.path.exists(modelFilePath):
            print('[FATAL] -- Output trained model file was not created.')
            exit(1)

        reprodOutputLogFile = os.path.join(modelOutputDir, f'pruneTune_{self.datasetName}_{date.today()}.log')
        reprodLog = list()
        reprodLog.append(f"{os.path.join(modelOutputDir, f'pruneTune_{self.datasetName}_{date.today()}.log')}")
        reprodLog.append(f"Pruning and finetuning completed on {time.ctime(time.time())}")
        reprodLog.append(f"Model saved to {modelOutputDir}")
        reprodLog.append(f"Pruning script run with the following command:")
        reprodLog.append("python3 " + " ".join(sys.argv))
        reprodLog.append(f"Process took {timeDelta:.2f} seconds with {lenLoss} epochs")
        reprodLog.append(f"Specific epochs per round: {finalEpochs}")
        reprodLog.append(f"Reached {ESSR} E.S.S.R. with the given pruning rounds:")
        reprodLog.append(f"Pruning rounds: {pruningRatesPerRound}")
        reprodLog.append(f"Time per round: {roundTimes}")
        # reprodLog.append(f"Final loss: {maxTrainLoss}")
        reprodLog.append(f" -------------- Current library versions: --------------")
        reprodLog.append(f"Script Version (SHA256): {hashlib.sha256(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), __file__), 'rb').read()).hexdigest()}")
        reprodLog.append(f"Python: {sys.version}")
        reprodLog.append(f"Tensorflow: {tf.__version__}")
        reprodLog.append(f"NVidia CUDA Runtime version: {build_info.build_info['cuda_version']}")
        reprodLog.append(f"NVidia CUDNN Runtime version: {build_info.build_info['cudnn_version']}")
        reprodLog.append(f" -------------- Current train object parameters: --------------")
        reprodLog.append(f"Model file export path: {modelFilePath}")
        reprodLog.append(f"Model hash (SHA256): {hashlib.sha256(open(modelFilePath, 'rb').read()).hexdigest()}")
        reprodLog.append(f"Dataset file: {self.inputTracesFile}")
        reprodLog.append(f"Dataset name: {self.datasetName}")
        reprodLog.append(f"Dataset file extension: {self.datasetFileExtension}")
        reprodLog.append(
            f"Dataset hash (SHA256): {hashlib.sha256(open(self.inputTracesFile, 'rb').read()).hexdigest()}")
        reprodLog.append(f"Train traces: {self.trainTracesNum}")
        reprodLog.append(f"Attack window: {self.attackWindow}")
        reprodLog.append(f"Target byte: {self.targetByte}")
        reprodLog.append(f"Leakage model: {self.leakageModel.value}")
        reprodLog.append(f"Verbose: {self.verbose}")
        reprodLog.append(f" -------------- Selected Log Items: --------------")
        reprodLog.extend(self.logs)
        reprodLog.append(f"")

        # Write each line to the log file
        with open(reprodOutputLogFile, 'w') as f:
            for line in reprodLog:
                f.write(f"{line}\n")

    def regenerateFPGML2Scores(self, model, i):
        """
        Regenerate FPGM or L2 scores for the model
        :param model: The Keras model to regenerate the scores for
        :param i: The current round number
        :return: None
        """
        scoreOutputDir = os.path.join(self.outputDir, f'Round{i}')
        if self.filterScore == "fpgm":
            FPGM_L2_Score.fpgm_scores(model, scoreOutputDir, False)
            self.ranksPath = os.path.join(scoreOutputDir, 'fpgm_idx.csv')
        else:
            FPGM_L2_Score.l2_scores(model, scoreOutputDir, False)
            self.ranksPath = os.path.join(scoreOutputDir, 'l2_idx.csv')

    def loadTrainingData(self):
        """
        Load the training data from the input traces npz or h5 file
        :return: The loaded traces, labels, and input shape
        """
        if self.datasetFileExtension == 'npz':  # Our format
            loadedNPZ = np.load(self.inputTracesFile)
            tmp = self.attackWindow.split('_')
            start_loc, end_loc = int(tmp[0]), int(tmp[1])
            parsedAttackWindow = [start_loc, end_loc]
            traces, labels, text_in, key, inp_shape = process_data.process_raw_data(loadedNPZ, self.targetByte,
                                                                                    self.leakageModel.value,
                                                                                    parsedAttackWindow)
            inp_shape = (traces.shape[1], 1)
            traces = traces[:self.trainTracesNum]
            labels = labels[:self.trainTracesNum]

        elif self.datasetFileExtension == 'h5' and self.datasetName == "ASCAD":  # ASCADv1 Format
            print("[WARNING] -- Loaded ASCADv1 database ignores attack window, target byte, preprocess and shifted")
            self.attackWindow = "0_700"
            self.targetByte = 2
            with h5py.File(self.inputTracesFile) as in_file:
                traces = np.array(in_file['Profiling_traces/traces'])[:self.trainTracesNum, :]
                labels = np.array(in_file['Profiling_traces/labels'], dtype='uint8')[:self.trainTracesNum]
                text_in = np.array(in_file['Profiling_traces/metadata'])['plaintext'][:self.trainTracesNum]
                key = np.array(in_file['Profiling_traces/metadata'])['key'][:self.trainTracesNum]
                inp_shape = (traces.shape[1], 1)

        elif self.datasetFileExtension == "h5" and self.datasetName == "ASCADv2":
            # TUPLE KEY: TracesX[0]: train, TracesX[1]: eval | TracesY[0]: train, TracesY[1]: eval | inp_shape
            traces, labels, inp_shape = ASCADv2Adapter.loadData(self.inputTracesFile, self.trainTracesNum,
                                                                self.valTracesNum, self.ascadV2Type)

        else:
            print('[FATAL] -- Dataset not recognized within loadTrainingData function.')
            exit(1)
        return traces, labels, inp_shape

    def applyPruningStrategyAndRun(self):
        """
        Main functionality of file.
        Prune / finetune the model given a specified pruning strategy, training data, and the passed parameters
        :return: None
        """
        # Load seed model (-m, --input_model_dir) and initial shape
        model = load_model(self.inputModelFile)
        input_layer_shape = model.get_layer(index=0).input_shape
        if isinstance(input_layer_shape, list):
            input_layer_shape = input_layer_shape[0]

        # Load traces and reshape for training based on architecture
        X_profiling, Y_profiling, input_shape = self.loadTrainingData()
        if self.architecture != "ASCADv2":
            Reshaped_X_profiling = process_data.sanity_check(input_layer_shape, X_profiling)
            Y_profiling = to_categorical(Y_profiling, self.leakageModel.getEmbeddingSize(self.leakageModel))
            Reshaped_X_profiling = Reshaped_X_profiling[:self.trainTracesNum]  # Ensure no more than specified traces
            Y_profiling = Y_profiling[:self.trainTracesNum]
        else:  # self.datasetName == "ASCADv2"  # Already reshape in the loadTrainingData for ASCADv2
            Reshaped_X_profiling = X_profiling
        initialModelFilterSizes = getModelSize(model)  # Extract initial shape from the passed model
        self.simpleLog(f"[LOG] -- With original model params {initialModelFilterSizes}", True)

        # 2D array of pruning ratios allow for individual pruning ratios per layer per round
        roundRatios = [[0] * len(initialModelFilterSizes)]
        epochsPerRound = list()

        # ---------------------------------- Iterative ----------------------------------
        if self.pruningType == 'iterative':
            # Automatic vars
            automaticPruningRate = 0.3
            decrementFactor = 0.1
            # self.maxEpochs = 650 if self.datasetName != "ASCADv2" else 50
            self.tryCounter = 20 if self.architecture != "ASCADv2" else 4
            self.stepSize = 25 if self.architecture != "ASCADv2" else 10
            roundRatios = [[1 - automaticPruningRate] * len(initialModelFilterSizes)]
            epochsPerRound = [int(self.epochs)]
            trainingTracesIndicesPerRound = [0, self.trainTracesNum - self.valTracesNum]

        # ------------------------------------ PTQ -------------------------------------
        # Apply PTQ then quit. TFLite models can only 'technically' be finetuned afterward
        elif self.pruningType == 'ptq':
            modelDir = os.path.join(self.outputDir, 'model')
            os.makedirs(modelDir, exist_ok=True)
            model_save_file = os.path.join(modelDir, 'best_model.tflite')
            tmp = self.attackWindow.split('_')
            poiWidth = int(tmp[1]) - int(tmp[0])
            model = model_zoo.applyPTQ(model, self.quantizationType, poiWidth, X_profiling)
            with open(model_save_file, 'wb') as f:
                f.write(model)
            self.simpleLog(f"Done! Model saved to {model_save_file}", True)
            exit(0)

        # ----------------------- Main prune+finetune loop start -----------------------
        # Start total timer
        totalTrainTimeStart = time.time()

        # Loop setup
        endCondition = len(roundRatios)
        i = 0
        tryCounter = 0

        # Statistic collection
        totalTrainingLoss = list()
        totalTrainingValLoss = list()
        finalEpochsPerRound = list()
        timePerRound = list()

        # Special Flags
        hasBumpedFilters = True  # Experimental option turned off by setting to true

        # --------------------------- Main prune+tune loop  ----------------------------
        roundTimeStart = time.time()
        while i < endCondition:
            gc.collect()  # Better stability over many rounds
            # Make empty model and copy over current weights from the seed model
            if self.architecture != "ASCADv2":
                if not self.fpgaArchitecture:
                    model_pruned = model_zoo.cnn_best(input_shape, roundRatios[i], initialSize=initialModelFilterSizes,
                                                  emb_size=self.leakageModel.getEmbeddingSize(self.leakageModel),
                                                  classification=True)
                else:
                    model_pruned = model_zoo.cnn_best_fpga2D(input_shape, roundRatios[i], initialSize=initialModelFilterSizes,
                                                      emb_size=self.leakageModel.getEmbeddingSize(self.leakageModel),
                                                      classification=True)
            else:  # For ASCADv2 model, use ESSR method of calculating intermediate filter sizes
                model_pruned = ASCADv2Adapter.getModel(self.ascadV2Type, (np.prod([x[0] for x in roundRatios])),
                                                       self.fpgaArchitecture)

            model_pruned = architectureExtractor.copyLayers(model, model_pruned, self.ranksPath)
            printRoundHeader(i, endCondition, 'automatic', roundRatios[i], epochsPerRound[i], getModelSize(model_pruned))

            modelDir = os.path.join(self.outputDir, f'Round{i}', 'model')
            os.makedirs(modelDir, exist_ok=True)
            model_save_file = os.path.join(modelDir, 'best_model.h5')
            history = train_model(Reshaped_X_profiling, Y_profiling, trainingTracesIndicesPerRound, model_pruned,
                                  model_save_file, epochsPerRound[i], verbose=self.verbose, datasetType=self.architecture)

            # ------------------------- Iterative Pruning --------------------------
            # ---------------------- Threshold Accuracy Chart ----------------------
            # 0.008 good for STM32 Power ID, STM32 EM ID, and STM32 Power D50 ID
            # 0.3 good for XMega Power HW, XMega EM HW, and XMega Power D50 HW
            # 0.006 good for FPGA Power ID, ASCAD
            # Uses dedicated validation slice of data (does NOT touch testing data)
            model_pruned = load_model(model_save_file)
            if self.architecture != "ASCADv2":
                _, accuracy = model_pruned.evaluate(Reshaped_X_profiling[-self.valTracesNum:],
                                                    Y_profiling[-self.valTracesNum:], verbose=False)
            else:
                testResults = model_pruned.evaluate(Reshaped_X_profiling[1], Y_profiling[1], verbose=False)
                accuracy = testResults[-32:-16] if self.ascadV2Type == "withPermIDs" else testResults[-16:]
                # Calculate model threshold accuracy for multiple attacked bytes, trying mean accuracy
                self.simpleLog(f"[LOG] -- *ASCADv2* All current accuracies: {accuracy}", True)
                accuracy = np.mean(accuracy)
                # accuracy = np.min(accuracy)
            self.simpleLog(f"[LOG] -- Current accuracy: {accuracy}", True)

            # --------------- Retrain Method of Iterative Pruning --------------
            #       If accuracy threshold is not met, finetune the model from the original round's
            #       base seed model with new epoch value for the full amount of epochs
            if accuracy < self.valAccuracy:
                tryCounter += 1
                self.simpleLog(f"Try counter: {tryCounter}", True)
                epochsPerRound[i] = epochsPerRound[i] + self.stepSize
                if tryCounter > self.tryCounter:
                    self.simpleLog(f"Model failed to meet threshold acc. after {tryCounter - 1} attempts, "
                                   f"adjusting ratio...", True)
                    tryCounter = 0
                    if automaticPruningRate - decrementFactor > 0.0001:  # Epsilon value
                        automaticPruningRate -= decrementFactor
                        epochsPerRound[i] = int(self.epochs)
                        roundRatios[i] = [1 - automaticPruningRate] * len(initialModelFilterSizes)

                        # Test for stopping criterion 1, cannot be reduced further
                        newEndCondition = stoppingCriteria1(getModelSize(model_pruned), roundRatios[i], endCondition)
                    else:  # Stopping criterion 2, minimum pruning rate failure
                        currentRoundDir, model_save_file, endCondition = stoppingCriteria2(self.outputDir, i, endCondition)
                        roundTimeEnd = time.time()
                        timePerRound.append(roundTimeEnd - roundTimeStart)
                continue

            else:  # Threshold accuracy is met. For 'retrain' method, we continue to the next round (if any)
                # In the automatic version of the pruning algorithm, need to lengthen training schedule if
                # neither of the stopping criteria have been met.
                #   1. Record statistics
                #   2. Reset tryCounter
                #   3. Extend endCondition, current pruning ratios, and starting epochs to the next round
                #   4. Perform a check to ensure that after application of pruning that the filter size's change
                self.simpleLog("[LOG] -- Acc. threshold met, continuing to next round...", True)
                totalTrainingLoss.extend(history.history['loss'])
                totalTrainingValLoss.extend(history.history['val_loss'])
                finalEpochsPerRound.append(epochsPerRound[i])
                tryCounter = 0
                endCondition += 1
                roundRatios.append(roundRatios[i])
                epochsPerRound.append(int(self.epochs))
                model = model_pruned
                initialModelFilterSizes = getModelSize(model_pruned)
                if checkIfFilterSizeSame(getModelSize(model_pruned), roundRatios[i]):  # Stopping criterion 1, filters unchanged
                    if hasBumpedFilters:
                        self.simpleLog(f"Stopping criterion reached, filters unchanged after pruning ratio applied.", True)
                        endCondition -= 1  # To let loop end naturally
                    else:
                        self.simpleLog("Bumping filters...", True)
                        hasBumpedFilters = True
                        roundRatios[-1] = [0.5] * len(roundRatios[i])
                        automaticPruningRate = 0.5

            # Regenerate FPGM or L2 scores for the new model, just saved in the same folder as the model
            self.regenerateFPGML2Scores(model, i)

            # End round timer
            roundTimeEnd = time.time()
            timePerRound.append(roundTimeEnd - roundTimeStart)

            # INCREMENT LOOP
            i += 1
            roundTimeStart = time.time()
        # ------------------------------------------- END PRUNE LOOP ---------------------------------------------------

        # End Total Timer
        totalTrainTimeEnd = time.time()
        totalTrainTime = totalTrainTimeEnd - totalTrainTimeStart

        # Create final graphs and print stats
        createLossGraph(self.outputDir, self.datasetName, 'pruning_loss', endCondition, totalTrainingLoss,
                        totalTrainingValLoss)
        genInfoPlots.createTimeGraph(timePerRound, "Rounds", "Time (s)", endCondition, self.datasetName,
                                     self.leakageModel.value, self.outputDir, "time_per_round", show=False)

        pruningRatePerRound = list()
        for ratioList in roundRatios:
            pruningRatePerRound.append(ratioList[0])
        reformattedPruningRates = ""
        for pruningRateR in pruningRatePerRound[:-1]:
            reformattedPruningRates += f"{pruningRateR:.1f} "
        roundRatios = roundRatios[:-1]
        essr = 1 - (np.prod([x[0] for x in roundRatios]))
        self.simpleLog(f"[LOG] -- Total training time: {totalTrainTime}", True)
        self.simpleLog(f"[LOG] -- Time per round: {timePerRound}", True)
        self.simpleLog(f"[LOG] -- Pruning ratios per round: {reformattedPruningRates}", True)
        self.simpleLog(f"[LOG] -- E.S.S.R.: {essr}", True)
        self.simpleLog(f"[LOG] -- Final Epochs per round: {finalEpochsPerRound}", True)
        self.simpleLog(f"[LOG] -- Exporting reproducibility log...", True)
        self.exportReproducibilityStats(totalTrainTime, timePerRound, reformattedPruningRates, essr, totalTrainingLoss,
                                        len(totalTrainingLoss), finalEpochsPerRound, self.outputDir, model_save_file)
        print("Done.")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_traces', help='Input traces to fine tune the model')
    parser.add_argument('-m', '--input_model_dir', help='Input original model to fine tune')
    parser.add_argument('-o', '--output_dir', help='Output directory for the fine tuned model')
    parser.add_argument('-e', '--epochs', help='Epochs to fine tune the model')
    parser.add_argument('-tb', '--target_byte', type=int, help='Target byte')
    parser.add_argument('-lm', '--network_type', choices={'HW', 'ID'}, help='Leakage model of the network')
    parser.add_argument('-arch', '--architecture', choices={'CNN', 'ASCADv2'}, help='Architecture of passed model')
    parser.add_argument('-aw', '--attack_window', default='', help='Attack window for the traces')
    parser.add_argument('-pt', '--pruning_type', choices={'iterative', 'ptq'}, help='Type of pruning to use')
    parser.add_argument('-tt', '--training_traces', type=int, help='Training traces to use')
    parser.add_argument('-vt', '--validation_traces', type=int, help='Number of validation traces for accuracy check')
    parser.add_argument('-va', '--validation_accuracy', type=float, help='Accuracy threshold during finetuning accuracy check')
    parser.add_argument('-fs', '--filter_score', choices={'l2', 'fpgm'}, help='Choice of model score algorithm')
    parser.add_argument('-qt', '--quantization_type', choices={'tfliteOnly', 'dynamicInt8', 'dynamicWRepInt8', 'float16', 'int8', 'uint8'}, help='Type of quantization to use')
    parser.add_argument('-rp', '--ranks_path', help='Path to L2 or FPGM IDX csv ranks file')
    parser.add_argument('-ascadV2Type', '--ascadV2Type', choices={"withPermIDs", "withoutPermIDs"}, help='Selects the type of ASCADv2 ResNet to generate')
    parser.add_argument('-fpga', '--fpga', action='store_true', help='Use FPGA compatible model architecture')
    parser.add_argument('-v', '--verbose', action='store_true', help='Include for verbose output')
    opts = parser.parse_args()
    return opts


# Main function
if __name__ == "__main__":
    cmdArgs = parseArgs()
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # Initialize the trainer and train the model
    trainer = SideChannelPrunerAndFinetuner(cmdArgs.input_traces, cmdArgs.input_model_dir, cmdArgs.output_dir,
                                            cmdArgs.epochs, cmdArgs.target_byte, cmdArgs.network_type,
                                            cmdArgs.architecture, cmdArgs.attack_window, cmdArgs.pruning_type,
                                            cmdArgs.training_traces, cmdArgs.validation_traces,
                                            cmdArgs.validation_accuracy, cmdArgs.filter_score,
                                            cmdArgs.quantization_type, cmdArgs.ranks_path, cmdArgs.ascadV2Type,
                                            cmdArgs.fpga, cmdArgs.verbose)
    trainer.applyPruningStrategyAndRun()
