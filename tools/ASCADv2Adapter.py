# ASCADv2Adapter.py - Start 7/12/24 - Logan Reichling
# This file contains all necessary code to iteratively prune ASCADv2 without making the rest of my code look too bad

from tools import ASCAD_train_models


def loadData(inputTracesFile, maxTrainTraces, numValTraces, ascadV2Type):
    """
    loadData - Load ASCADv2 data and return it in a format that is familar to the rest of the code
    :param inputTracesFile: The filepath to the ASCADv2 data
    :param maxTrainTraces: Maximum number of traces to use for training
    :param numValTraces: Number of traces to use for validation accuracy calculation
    :param ascadV2Type: Type of ASCADv2 ResNet model to train
    :return: Tuple of training traces [train, eval], Tuple of validation traces [train, eval], and input shape
    """
    (X_profiling, Y_profiling), (X_attack, Y_attack), (profilingMetadata, attackMetadata) \
        = ASCAD_train_models.load_ascad(inputTracesFile, True)
    X_profiling_train = X_profiling[:(maxTrainTraces - numValTraces)]
    Y_profiling_train = Y_profiling[:(maxTrainTraces - numValTraces)]
    Reshaped_X_profiling_train = X_profiling_train.reshape((X_profiling_train.shape[0], X_profiling_train.shape[1], 1))
    if ascadV2Type == "withPermIDs":
        Reshaped_Y_profiling_train = ASCAD_train_models.multilabel_to_categorical(Y_profiling_train)
    else:  # ascadV2Type == "withoutPermIDs":
        Reshaped_Y_profiling_train = ASCAD_train_models.multilabel_without_permind_to_categorical(Y_profiling_train)
    X_profiling_eval = X_profiling[(maxTrainTraces - numValTraces):maxTrainTraces]
    Y_profiling_eval = Y_profiling[(maxTrainTraces - numValTraces):maxTrainTraces]
    Reshaped_X_profiling_eval = X_profiling_eval.reshape((X_profiling_eval.shape[0], X_profiling_eval.shape[1], 1))
    if ascadV2Type == "withPermIDs":
        Reshaped_Y_profiling_eval = ASCAD_train_models.multilabel_to_categorical(Y_profiling_eval)
    else:  # ascadV2Type == "withoutPermIDs":
        Reshaped_Y_profiling_eval = ASCAD_train_models.multilabel_without_permind_to_categorical(Y_profiling_eval)

    # Traces, labels, inp_shape
    return ((Reshaped_X_profiling_train, Reshaped_X_profiling_eval),
            (Reshaped_Y_profiling_train, Reshaped_Y_profiling_eval),
            (X_profiling_train.shape[0], X_profiling_train.shape[1], 1))


def getModel(ascadV2Type: str, ratio: float, is2DMode: bool):
    if is2DMode:
        if ascadV2Type == "withPermIDs":
            return ASCAD_train_models.resnet_v1_2d((15000, 1, 1), 19, ratio=ratio)
        else:  # ascadV2Type == "withoutPermIDs":
            return ASCAD_train_models.resnet_v1_2d((15000, 1, 1), 19, without_permind=1, ratio=ratio)
    else:
        if ascadV2Type == "withPermIDs":
            return ASCAD_train_models.resnet_v1((15000, 1), 19, ratio=ratio)
        else:  # ascadV2Type == "withoutPermIDs":
            return ASCAD_train_models.resnet_v1((15000, 1), 19, without_permind=1, ratio=ratio)

