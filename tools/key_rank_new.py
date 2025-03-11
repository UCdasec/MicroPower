# key_rank_new.py - Refactor start 12/30/24 - UC DASEC - Logan Reichling
# Computes the GE / MTD data for side-channel attack predictions (in parallel as of 1/2/25)
import os
import random
import secrets
from os import cpu_count
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tools.SideChannelConstants import SideChannelConstants
from tools import genInfoPlots
from multiprocessing import Process, Queue

def computeRankingCurve(preds, key, plaintext, targetByte, rankRoot, leakageModel, traceNumMax, fileName):
    """
    Parallelized version of the ranking curve computation
    :param preds:        List of classifier predictions
    :param key:          Full key for the target dataset
    :param plaintext:    Full plaintext list for the target dataset
    :param targetByte:   Targeted byte for the ranking curve
    :param rankRoot:     Filepath to save the ranking curve
    :param leakageModel: Leakage model identifier
    :param traceNumMax:  Number of traces to use for the ranking curve
    :param fileName:     Filename suffix for the ranking curve output files
    :return: None
    """
    numberOfAttacksToAverage = 100  # Average GE/KR over 100 attacks (permutations of traces)
    aesSBox      = SideChannelConstants.getAESSBox()
    hwConversion = SideChannelConstants.getHWSBoxConversion()
    hwSetLengths = SideChannelConstants.getHWSetLengths()  # Speed up HW LM a bit
    finalGEs     = np.zeros((numberOfAttacksToAverage, traceNumMax))
    realKey     = key[targetByte]
    plaintext    = plaintext[:, targetByte]
    resultQueue  = Queue()
    progress     = tqdm(total=numberOfAttacksToAverage)
    processes    = list()
    jobIndex     = 0
    bytesPerSeed = 32
    randomBytes  = secrets.token_bytes(bytesPerSeed * numberOfAttacksToAverage)
    for proc in range(numberOfAttacksToAverage):
        processes.append(Process(target=singleRankingCurve, args=(preds, realKey, plaintext, leakageModel, traceNumMax,
                          aesSBox, hwConversion, hwSetLengths, resultQueue,
                          randomBytes[0+proc*bytesPerSeed:(bytesPerSeed-1)+proc*bytesPerSeed]), daemon=False))
    for i in range(cpu_count()):  # Ease memory usage for increased speed up
        processes[jobIndex].start()
        jobIndex += 1
    for i in range(numberOfAttacksToAverage):
        finalGEs[i, :] = resultQueue.get()
        progress.update(1)
        if jobIndex < numberOfAttacksToAverage:
            processes[jobIndex].start()
            jobIndex += 1
    for process in processes:
        process.join()
    progress.close()
    finalGEs = np.mean(finalGEs, axis=0)  # Average the attacks

    # Create result directory and save the result plots
    os.makedirs(rankRoot, exist_ok=True)
    genInfoPlots.plotDefaultRankCurve(finalGEs[0:traceNumMax], targetByte, leakageModel)
    figureSavePath = os.path.join(rankRoot, 'ranking_curve.png')
    plt.savefig(figureSavePath)
    plt.clf()
    data = finalGEs[0:traceNumMax]
    firstBecomeZero, lastBecomeZero = genInfoPlots.findZeroIndices(data)   # For additional zoomed plot
    if lastBecomeZero != -1:
        croppedData = genInfoPlots.create_cropped_array(data, lastBecomeZero)  # Crop original dataset
    elif firstBecomeZero != -1:
        croppedData = genInfoPlots.create_cropped_array(data, firstBecomeZero)
    else:
        croppedData = data
    genInfoPlots.plotKeyRankCurve(croppedData, f"{targetByte}", "Number of Traces", "Key Rank",
                                  "TEST", f"{leakageModel}", firstBecomeZero, lastBecomeZero, show=True)
    figureSavePath = str(os.path.join(rankRoot, "key_rank_cropped_" + fileName + ".png"))
    plt.savefig(figureSavePath)
    print(f'[LOG] -- Ranking curves saved to path: {figureSavePath}')

    # Save the raw ranking data
    rawDataSavePath = str(os.path.join(rankRoot, 'ranking_raw_data_' + fileName + '.npz'))
    x = list(range(len(finalGEs)))
    np.savez(rawDataSavePath, x=x, y=finalGEs)
    print(f'[LOG] -- Raw ranking data saved to path: {rawDataSavePath}')


def singleRankingCurve(preds, realKey, plaintext, leakageModel, traceNumMax, aesSBox, hwConversion, hwSetLengths,
                       resultQueue, seed=None):
    """
    Compute MTD results with a single shuffle of the pre-generated predictions for use in parallel computation
    :param preds:         List of classifier predictions
    :param realKey:       Ground truth key byte value
    :param plaintext:     List of plaintext bytes for each accompanying prediction
    :param leakageModel:  String identifier Leakage model
    :param traceNumMax:   Number of traces leveraged
    :param aesSBox:       AES S-Box constant
    :param hwConversion:  HW Conversion List constant
    :param hwSetLengths:  HW Set Length List constant
    :param resultQueue:   Output queue to place results
    :param seed:          Bytes to initialize random seed
    :return: None
    """
    random_index = list(range(plaintext.shape[0]))
    random.seed(seed)
    random.shuffle(random_index)
    random_index = random_index[0:traceNumMax]
    singleGuessingEntropy = np.zeros(traceNumMax)
    score_mat = np.zeros((traceNumMax, 256))
    for key_guess in range(0, 256):
        for i in range(0, traceNumMax):
            initialState = int(plaintext[random_index[i]]) ^ key_guess
            sout = aesSBox[initialState]
            if leakageModel == 'ID':
                label = sout
                score_mat[i, key_guess] = preds[random_index[i], label]
            elif leakageModel == 'HW':
                label = hwConversion[sout]
                prob_value_share = preds[random_index[i], label] / hwSetLengths[label]
                score_mat[i, key_guess] = prob_value_share
    score_mat = np.log(score_mat + 1e-40)
    for i in range(0, traceNumMax):
        log_likelihood = np.sum(score_mat[0:i + 1, :], axis=0)
        ranked = list(np.argsort(log_likelihood)[::-1])
        singleGuessingEntropy[i] = ranked.index(realKey)
    resultQueue.put(singleGuessingEntropy)


def ascadV2RankingCurves(allKeyRankArrays, ascadV2Type, outputDir):
    """
    Create several ranking curves for the ASCADv2 dataset, including the key rank curve for all individual keys
    :param allKeyRankArrays: All key rank arrays, x_i and y_i per indices
    :param ascadV2Type:      With or without PermIDs
    :param outputDir:        The output directory for the plots
    :return: None
    """
    mostTracesNeeded = -1
    maxFirstValue = -1
    for keyRank in allKeyRankArrays:
        if keyRank[1][0] > maxFirstValue:
            maxFirstValue = keyRank[1][0]
        bytePerformance = genInfoPlots.findZeroIndices(keyRank[1])[0]
        if bytePerformance > mostTracesNeeded:
            mostTracesNeeded = bytePerformance
    for i in range(16):
        plt.plot(allKeyRankArrays[i][0], allKeyRankArrays[i][1], label="Key_" + str(i))
    if mostTracesNeeded != -1:
        plt.axvline(x=mostTracesNeeded, color='#00CC33', linestyle="--")
        plt.annotate(f"Last key converge at {mostTracesNeeded}", (mostTracesNeeded, maxFirstValue / 2),
                     xytext=(len(allKeyRankArrays[0][0])*0.3, maxFirstValue * 0.925),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    plt.title('Performance of MultiSCAResNet\n'+("(w/ PermIDs)" if ascadV2Type == "withPermIDs" else "(w/o PermIDs") +
              ' against ASCADv2')
    plt.xlabel('Traces')
    plt.ylabel('Rank')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(outputDir, "key_rank_multiSCA.png"))
    plt.savefig(os.path.join(outputDir, "key_rank_multiSCA.svg"))
    plt.show(block=False)
    plt.clf()

    # Close up graph of each byte as well:
    for i in range(16):
        data = allKeyRankArrays[i][1]
        first_become_zero, last_become_zero = genInfoPlots.findZeroIndices(data)  # Find important indices
        if last_become_zero != -1:
            cropped_data = genInfoPlots.create_cropped_array(data, last_become_zero)  # Crop original dataset
        elif first_become_zero != -1:
            cropped_data = genInfoPlots.create_cropped_array(data, first_become_zero)
        else:
            cropped_data = data
        genInfoPlots.plotKeyRankCurve(cropped_data, f"{i}", "Number of Traces", "Key Rank",
                                      f"MultiSCA ASCADv2 {ascadV2Type}", f"ID", first_become_zero, last_become_zero,
                                      show=False)  # Plot the rank curve
        plt.savefig(os.path.join(outputDir, f"key_rank_cropped_{i}_multiSCA.png"))
        plt.savefig(os.path.join(outputDir, f"key_rank_cropped_{i}_multiSCA.svg"))
        plt.clf()

# Original ranking_curve
# def computeRankingCurveOriginal(preds, key, plaintext, target_byte, rank_root, leakage_model, trace_num_max, fileName):
#     """
#     - preds : the probability for each class (n*256 for a byte, n*9 for Hamming weight)
#     - real_key : the key of the target device
#     - device_id : id of the target device
#     - model_flag : a string for naming GE result
#     """
#     num_averaged = 100  # GE/SR is averaged over 100 attacks
#     aesSBox      = SideChannelConstants.getAESSBox()
#     hwConversion = SideChannelConstants.getHWSBoxConversion()
#     hwSetLengths = SideChannelConstants.getHWSetLengths()  # Speed up HW LM a bit
#     guessing_entropy = np.zeros((num_averaged, trace_num_max))  # max trace num for attack
#     real_key = key[target_byte]
#     plaintext = plaintext[:, target_byte]
#     # success_flag = np.zeros((num_averaged, trace_num_max))
#
#     for time in tqdm(range(num_averaged)):
#         random_index = list(range(plaintext.shape[0]))
#         random.shuffle(random_index)
#         random_index = random_index[0:trace_num_max]
#         # initialize score matrix
#         score_mat = np.zeros((trace_num_max, 256))
#         for key_guess in range(0, 256):
#             for i in range(0, trace_num_max):
#                 initialState = int(plaintext[random_index[i]]) ^ key_guess
#                 sout = aesSBox[initialState]
#                 if leakage_model == 'ID':
#                     label = sout
#                     score_mat[i, key_guess] = preds[random_index[i], label]
#                 elif leakage_model == 'HW':
#                     label = hwConversion[sout]
#                     prob_value_share = preds[random_index[i], label] / hwSetLengths[label]
#                     score_mat[i, key_guess] = prob_value_share
#         score_mat = np.log(score_mat + 1e-40)
#         for i in range(0, trace_num_max):
#             log_likelihood = np.sum(score_mat[0:i + 1, :], axis=0)
#             ranked = list(np.argsort(log_likelihood)[::-1])
#             guessing_entropy[time, i] = ranked.index(real_key)
#             # if ranked.index(real_key) == 0:
#             #     success_flag[time, i] = 1
#
#     # maxGEs = np.max(guessing_entropy, axis=0)
#     # minGEs = np.min(guessing_entropy, axis=0)
#     for row in range(num_averaged):
#         print(guessing_entropy[row][0:5])
#
#     guessing_entropy = np.mean(guessing_entropy, axis=0)  # Average the attacks
#
#     # define the saving path
#     os.makedirs(rank_root, exist_ok=True)
#
#     # only plot guess entry
#     plt.figure(figsize=(8, 6))
#     plt.plot(guessing_entropy[0:trace_num_max], color='red')
#     plt.title('Leakage model: {}, target byte: {}'.format(leakage_model, target_byte))
#     plt.xlabel('Number of trace')
#     plt.ylabel('Key Rank')
#     fig_save_path = os.path.join(rank_root, 'ranking_curve.png')
#     plt.savefig(fig_save_path)
#     plt.show()
#     plt.close()
#     print('[LOG] -- ranking curve save to path: ', fig_save_path)
#
#     # saving the ranking raw data
#     raw_save_path = os.path.join(rank_root, 'ranking_raw_data_' + fileName + '.npz')
#     x = list(range(len(guessing_entropy)))
#     np.savez(raw_save_path, x=x, y=guessing_entropy)
#     print('[LOG] -- ranking raw data save to path: ', raw_save_path)
#
#     # Additional logan plot
#     data = guessing_entropy[0:trace_num_max]
#     first_become_zero, last_become_zero = genInfoPlots.findZeroIndices(data)  # Find important indices
#     if last_become_zero != -1:
#         cropped_data = genInfoPlots.create_cropped_array(data, last_become_zero)  # Crop original dataset
#     elif first_become_zero != -1:
#         cropped_data = genInfoPlots.create_cropped_array(data, first_become_zero)
#     else:
#         cropped_data = data
#     genInfoPlots.plotKeyRankCurve(cropped_data, f"{target_byte}", "Number of Traces",
#                                   "Key Rank", "TEST", f"{leakage_model}",
#                                   first_become_zero, last_become_zero, show=True)  # Plot the rank curve
#     fig_save_path = os.path.join(rank_root, "key_rank_cropped_" + fileName + ".png")
#     plt.savefig(fig_save_path)
