# architectureExtractor.py - Start 6/23/24 - Logan Reichling
# The purpose of this file is to contain all functions dealing with low-level model weights and biases

import os
import pandas as pd
import numpy as np
from tools import model_zoo

# Tensorflow Specific (+ turn off their logging)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D, Add, Dense, Flatten
from tensorflow.keras.models import load_model
from tools import ASCADv2Adapter
tf.get_logger().setLevel('ERROR')


# def createFreshModelFromPretrained(trainedModel):
#     """
#     Creates a fresh model with the same architecture as the trained model, but with untrained weights.
#     :param trainedModel: The model with trained weights to copy the architecture from
#     :return: A fresh model with the same architecture as the trained model
#     """
#     fresh

def copyLayers(trainedModel, newModel, ranksPath, copyDense=False):
    """
    Copies convolutional layer weights from a trained model to a new model based on FPGM/L2 ranks.
    :param trainedModel: Model with trained weights
    :param newModel: New model with same architecture as trained model
    :param ranksPath: File path to FPGM/L2 ID ranks csv file
    :param copyDense: Experimental boolean to also copy over the dense layers
    :return: New model with copied weights
    """
    # Parse layer filter score rank ID file
    layerRankIDList = pd.read_csv(ranksPath, header=None).values
    totalFilterRanksPerLayer = list()
    for layerRankIDs in layerRankIDList:
        layerRankIDs = layerRankIDs[~np.isnan(layerRankIDs)]
        layerRankIDs = list(map(int, layerRankIDs))
        totalFilterRanksPerLayer.append(layerRankIDs[1:])  # Remove preceding conv. layer count

    # Extract layer objects from both models
    trainedConvLayers = list()
    trainedDenseLayers = list()
    skipBranchConvLayers = list()
    for i, layer in enumerate(trainedModel.layers):
        if type(layer) in [Conv1D, Conv2D]:
            trainedConvLayers.append(layer)
        elif type(layer) is Dense:
            trainedDenseLayers.append(layer)
        elif type(layer) is Add:  # Need to handle ResNet skip branches differently, extract automatically from add layers
            if tf.keras.__version__[0] != "3":
                for inbound_node in layer.inbound_nodes:  # For each item in the ADD layer
                    addedLayers = [tensor._keras_history.layer for tensor in inbound_node.input_tensors]  # Undocumented
                    for addLayer in addedLayers:
                        if type(addLayer) in [Conv1D, Conv2D]:
                            skipBranchConvLayers.append(addLayer)
            else:
                for inbound_node in layer._inbound_nodes:  # For each item in the ADD layer
                    addedLayers = [tensor._keras_history.operation for tensor in inbound_node.input_tensors]  # Undocumented
                    for addLayer in addedLayers:
                        if type(addLayer) in [Conv1D, Conv2D]:
                            skipBranchConvLayers.append(addLayer)


    newConvLayers = list()
    newDenseLayers = list()
    newFlattenOutputShape = 0
    newlyFlattenDenseLayers = list()
    for i, layer in enumerate(newModel.layers):
        if type(layer) in [Conv1D, Conv2D]:
            newConvLayers.append(layer)
        elif type(layer) is Dense:
            newDenseLayers.append(layer)
        elif type(layer) is Flatten:  # Need to extract output shape of flatten for immediately following dense layers
            newFlattenOutputShape = layer.output.shape[1]


    # Preliminary model length sanity check
    exitFlag = False
    # if len(totalFilterRanksPerLayer) != len(trainedConvLayers):
    #     print(f"[FATAL] -- FPGM/L2 ranks do not match loaded model!")
    #     exitFlag = True
    if len(trainedConvLayers) != len(newConvLayers):
        print(f"[FATAL] -- New model structure doesn't match existing model structure!")
        exitFlag = True
    if len(trainedDenseLayers) != len(newDenseLayers):
        print(f"[FATAL] -- New model structure doesn't match existing model structure!")
        exitFlag = True
    if exitFlag:
        exit(1)

    # Given new model size, copy conv. filters over from the trained model
    for i, layer in enumerate(newConvLayers):
        # print(f"{i}: {layer.filters}: {totalFilterRanksPerLayer[i][0:layer.filters]}")

        outputFiltersToCopyIndices = totalFilterRanksPerLayer[i][0:layer.filters]
        if i == 0:  # Later layers need to have input tensors set as well, can skip for input conv
            trainedWeights = trainedConvLayers[i].get_weights()[0][:, :, outputFiltersToCopyIndices] if type(layer) == Conv1D else trainedConvLayers[i].get_weights()[0][:, :, :, outputFiltersToCopyIndices]
        else:
            if trainedConvLayers[i] in skipBranchConvLayers:  # Can make this more generic in the future (no -3)
                inputFiltersToCopyIndices = totalFilterRanksPerLayer[i - 3][0:newConvLayers[i - 3].filters]
            else:
                inputFiltersToCopyIndices = totalFilterRanksPerLayer[i - 1][0:newConvLayers[i - 1].filters]
            trainedWeights = trainedConvLayers[i].get_weights()[0][:, :, outputFiltersToCopyIndices] if type(layer) == Conv1D else trainedConvLayers[i].get_weights()[0][:, :, :, outputFiltersToCopyIndices]
            trainedWeights = trainedWeights[:, inputFiltersToCopyIndices, :] if type(layer) == Conv1D else trainedWeights[:, :, inputFiltersToCopyIndices, :]
        trainedBiases = trainedConvLayers[i].get_weights()[1][outputFiltersToCopyIndices]
        layer.set_weights([trainedWeights, trainedBiases])

    # Copy Dense layers over if specified
    if copyDense:
        # Need to remove conv. layers from rank list, keep this line below for now....
        totalFilterRanksPerLayer = totalFilterRanksPerLayer[len(newConvLayers):]
        for i, layer in enumerate(newDenseLayers):
            # print("Name:", layer.name)
            # print(f"{i}: {layer.filters}: {totalFilterRanksPerLayer[i][0:layer.filters]}")
            outputFiltersToCopyIndices = totalFilterRanksPerLayer[i][0:layer.units]
            #print("Layer units", layer.units)
            #print("outputIndices", outputFiltersToCopyIndices)
            if i == 0:  # Set input to dense layer as Flatten output shape,
                trainedWeights = trainedDenseLayers[i].get_weights()[0][:, outputFiltersToCopyIndices]
                trainedWeights = trainedWeights[:newFlattenOutputShape, :]
            else:
                inputFiltersToCopyIndices = totalFilterRanksPerLayer[i - 1][0:newDenseLayers[i - 1].units]
                trainedWeights = trainedDenseLayers[i].get_weights()[0][:, outputFiltersToCopyIndices]
                trainedWeights = trainedWeights[inputFiltersToCopyIndices, :]
            trainedBiases = trainedDenseLayers[i].get_weights()[1][outputFiltersToCopyIndices]
            layer.set_weights([trainedWeights, trainedBiases])
    return newModel

if __name__ == "__main__":
    modelFile = load_model(r"C:\Users\Logan Reichling\Desktop\FOR_LOGAN\xmega_conv2d\model\best_model.h5")

    # print(modelFile.layers)
    # tempList = list()
    # for i, layer in enumerate(modelFile.layers):
    #     if type(layer) is Conv2D:
    #         tempList.append(layer)
    # print(tempList[0].get_weights()[0])
    # newArray = tempList[0].get_weights()[0].reshape((11,1,64))
    # print(newArray)
    # print(newArray.reshape((11,1,1,64)))

    print(modelFile.summary())
    new_model = model_zoo.cnn_best_fpga2D((1000, 1), emb_size=256, ratio=[0.5]*7)
    # new_model = ASCADv2Adapter.getModel("withoutPermIDs", 0.5)
    print(new_model.summary())
    ranks_path = r"C:\Users\Logan Reichling\Desktop\FOR_LOGAN\xmega_conv2d\l2_idx.csv"
    new_model = copyLayers(modelFile, new_model, ranks_path)
    print(new_model.summary())

