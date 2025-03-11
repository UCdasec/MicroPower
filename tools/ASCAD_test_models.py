# ASCAD_test_models.py - Modified 6/22/24 - Logan Reichling
# ASCAD_test_models.py original source available at https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_test_models.py
# Cite: L. Masure and R. Strullu, ‘Side Channel Analysis against the ANSSI’s protected AES implementation on ARM’. 2021.

import os
import random
import matplotlib as mpl
from tqdm import tqdm
if os.environ.get('DISPLAY', '') == '':
    # print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
else:
    mpl.use('TkAgg')
import os.path
import h5py
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model
tf.get_logger().setLevel('ERROR')
from tools.SideChannelConstants import SideChannelConstants

# Two Tables to process a field multplication over GF(256): a*b = alog (log(a) + log(b) mod 255)
log_table = [0, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3,
             100, 4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193,
             125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9, 120,
             101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218, 142,
             150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70, 131, 56,
             102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34, 136, 145, 16,
             126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84, 250, 133, 61, 186,
             43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172, 229, 243, 115, 167, 87,
             175, 88, 168, 80, 244, 234, 214, 116, 79, 174, 233, 213, 231, 230, 173, 232,
             44, 215, 117, 122, 235, 22, 11, 245, 89, 203, 95, 176, 156, 169, 81, 160,
             127, 12, 246, 111, 23, 196, 73, 236, 216, 67, 31, 45, 164, 118, 123, 183,
             204, 187, 62, 90, 251, 96, 177, 134, 59, 82, 161, 108, 170, 85, 41, 157,
             151, 178, 135, 144, 97, 190, 220, 252, 188, 149, 207, 205, 55, 63, 91, 209,
             83, 57, 132, 60, 65, 162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171,
             68, 17, 146, 217, 35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165,
             103, 74, 237, 222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7]

alog_table = [1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53,
              95, 225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170,
              229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230, 49,
              83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110, 178, 205,
              76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24, 40, 120, 136,
              131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206, 73, 219, 118, 154,
              181, 196, 87, 249, 16, 48, 80, 240, 11, 29, 39, 105, 187, 214, 97, 163,
              254, 25, 43, 125, 135, 146, 173, 236, 47, 113, 147, 174, 233, 32, 96, 160,
              251, 22, 58, 78, 210, 109, 183, 194, 93, 231, 50, 86, 250, 21, 63, 65,
              195, 94, 226, 61, 71, 201, 64, 192, 91, 237, 44, 116, 156, 191, 218, 117,
              159, 186, 213, 100, 172, 239, 42, 126, 130, 157, 188, 223, 122, 142, 137, 128,
              155, 182, 193, 88, 232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84,
              252, 31, 33, 99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202,
              69, 207, 74, 222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14,
              18, 54, 90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23,
              57, 75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1]


# Multiplication function in GF(2^8)
def multGF256(a, b):
    if (a == 0) or (b == 0):
        return 0
    else:
        return alog_table[(log_table[a] + log_table[b]) % 255]


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        exit(1)
    return


def load_sca_model(model_file):
    check_file_exists(model_file)
    try:
        model = load_model(model_file)
    except:
        print("Error: can't load Keras model file '%s'" % model_file)
        exit(1)
    return model


# Compute the rank of the real key for a give set of predictions
def rank(predictions, metadata, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte,
         simulated_key):
    # Compute the rank
    if len(last_key_bytes_proba) == 0:
        key_bytes_proba = np.zeros(256)  # initialize all the estimates to zero
    else:
        key_bytes_proba = last_key_bytes_proba  # using the previous computations to save time!
    aesSBox = SideChannelConstants.getAESSBox()
    for p in range(0, max_trace_idx - min_trace_idx):
        # Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
        plaintext = metadata[min_trace_idx + p]['plaintext'][target_byte]
        key = metadata[min_trace_idx + p]['key'][target_byte]
        for i in range(0, 256):
            # Our candidate key byte probability is the sum of the predictions logs
            if simulated_key != 1:
                proba = predictions[p][aesSBox[plaintext ^ i]]
            else:
                proba = predictions[p][aesSBox[plaintext ^ key ^ i]]
            if proba != 0:
                key_bytes_proba[i] += np.log(proba)
            else:
                # We do not want an -inf here, put a small epsilon that corresponds to a power of our min non zero proba
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes ... this should not happen!")
                    exit(1)
                min_proba = min(min_proba_predictions)
                key_bytes_proba[i] += np.log(min_proba ** 2)
    # Now we find where our real key candidate lies in the estimation.
    # We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a: key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return real_key_rank, key_bytes_proba


def full_ranks(predictions, dataset, metadata, min_trace_idx, max_trace_idx, rank_step, target_byte, simulated_key):
    print("Computing rank for targeted byte {}".format(target_byte))
    real_key = metadata[0]['key'][target_byte] if (simulated_key != 1) else 0  # Set simulated or real key

    # Check for overflow
    if max_trace_idx > dataset.shape[0]:
        print("Error: asked trace index %d overflows the total traces number %d" % (max_trace_idx, dataset.shape[0]))
        exit(1)

    # Average over NUM_AVERAGE runs
    NUM_AVERAGED = 100
    index = np.arange(min_trace_idx + rank_step, max_trace_idx + 1, rank_step)
    allFRanks = np.zeros((NUM_AVERAGED, len(index)))
    for run in tqdm(range(NUM_AVERAGED)):
        random.shuffle(index)
        f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
        key_bytes_proba = []
        for t, i in zip(index, range(0, len(index))):
            real_key_rank, key_bytes_proba = rank(predictions[t - rank_step:t], metadata, real_key, t - rank_step, t,
                                                  key_bytes_proba, target_byte, simulated_key)
            f_ranks[i] = [t - min_trace_idx, real_key_rank]
        allFRanks[run] = f_ranks[:, 1]
    allFRanks = np.mean(allFRanks, axis=0)
    xAxisValues = np.arange(min_trace_idx + rank_step, max_trace_idx + 1, rank_step)
    byteResults = np.zeros((max_trace_idx, 2))
    byteResults[:, 1] = allFRanks
    byteResults[:, 0] = xAxisValues
    return byteResults


#### ASCAD helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the ASCAD
# database
def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        exit(1)
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (
            in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])


# Compute Pr(Sbox(p^k)*alpha|t)
def proba_dissect_beta(proba_sboxmuladd, proba_beta):
    proba = np.zeros(proba_sboxmuladd.shape)
    for j in range(proba_beta.shape[1]):
        proba_sboxdeadd = proba_sboxmuladd[:, [(beta ^ j) for beta in range(256)]]
        proba[:, j] = np.sum(proba_sboxdeadd * proba_beta, axis=1)
    return proba


# Compute Pr(Sbox(p^k)|t)
def proba_dissect_alpha(proba_sboxmul, proba_alpha):
    proba = np.zeros(proba_sboxmul.shape)
    for j in range(proba_alpha.shape[1]):
        proba_sboxdemul = proba_sboxmul[:, [multGF256(alpha, j) for alpha in range(256)]]
        proba[:, j] = np.sum(proba_sboxdemul * proba_alpha, axis=1)
    return proba


# Compute Pr(Sbox(p[permind]^k[permind])|t)
def proba_dissect_permind(proba_x, proba_permind, j):
    proba = np.zeros((proba_x.shape[0], proba_x.shape[2]))
    for s in range(proba_x.shape[2]):
        proba_1 = proba_x[:, :, s]
        proba_2 = proba_permind[:, :, j]
        proba[:, s] = np.sum(proba_1 * proba_2, axis=1)
    return proba


# Compute Pr(Sbox(p^k)|t) by a recombination of the guessed probilities, with the permIndices known during the profiling phase
def multilabel_predict(predictions):
    predictions_alpha = predictions[0]
    predictions_beta = predictions[1]
    predictions_unshuffledsboxmuladd = []
    predictions_permind = []
    for i in range(16):
        predictions_unshuffledsboxmuladd.append(predictions[2 + i])
        predictions_permind.append(predictions[2 + 16 + i])

    predictions_unshuffledsboxmul = []
    print("Computing multiplicative masked sbox probas with shuffle...")
    for i in range(16):
        predictions_unshuffledsboxmul.append(proba_dissect_beta(predictions_unshuffledsboxmuladd[i], predictions_beta))

    print("Computing sbox probas with shuffle...")
    predictions_unshuffledsbox = []
    for i in range(16):
        predictions_unshuffledsbox.append(proba_dissect_alpha(predictions_unshuffledsboxmul[i], predictions_alpha))

    predictions_unshuffledsbox_v = np.array(predictions_unshuffledsbox)
    predictions_permind_v = np.array(predictions_permind)
    predictions_unshuffledsbox_v = np.moveaxis(predictions_unshuffledsbox_v, [0, 1, 2], [1, 0, 2])
    predictions_permind_v = np.moveaxis(predictions_permind_v, [0, 1, 2], [1, 0, 2])
    predictions_sbox = []
    print("Computing sbox probas...")
    for i in range(16):
        predictions_sbox.append(proba_dissect_permind(predictions_unshuffledsbox_v, predictions_permind_v, i))

    return predictions_sbox


# Compute Pr(Sbox(p^k)|t) by a recombination of the guessed probilities without taking the shuffling into account
def multilabel_without_permind_predict(predictions):
    predictions_alpha = predictions[0]
    predictions_beta = predictions[1]
    predictions_sboxmuladd = []
    for i in range(16):
        predictions_sboxmuladd.append(predictions[2 + i])

    predictions_sboxmul = []
    print("Computing multiplicative masked sbox...")
    for i in range(16):
        predictions_sboxmul.append(proba_dissect_beta(predictions_sboxmuladd[i], predictions_beta))

    print("Computing sbox probas...")
    predictions_sbox = []
    for i in range(16):
        predictions_sbox.append(proba_dissect_alpha(predictions_sboxmul[i], predictions_alpha))

    return predictions_sbox


# Check a saved model against one of the ASCAD databases Attack traces
def check_model(model_file, ascad_database, num_traces=2000, target_byte=2, multilabel=0, simulated_key=0,
                save_file=""):
    check_file_exists(model_file)
    check_file_exists(ascad_database)
    # Load profiling and attack data and metadata from the ASCAD database
    (X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_ascad(ascad_database,
                                                                                                         load_metadata=True)
    # Load model
    model = load_sca_model(model_file)
    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape[0]
    if isinstance(model.get_layer(index=0).input_shape, list):
        input_layer_shape = model.get_layer(index=0).input_shape[0]
    else:
        input_layer_shape = model.get_layer(index=0).input_shape
    # Sanity check
    if input_layer_shape[1] != len(X_attack[0, :]):
        print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_attack[0, :])))
        exit(1)
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        input_data = X_attack[:num_traces, :]
    elif len(input_layer_shape) == 3:
        # This is a CNN: reshape the data
        input_data = X_attack[:num_traces, :]
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        exit(1)
    # Predict our probabilities
    predictions = model.predict(input_data)
    if multilabel != 0:
        if multilabel == 1:
            predictions_sbox = multilabel_predict(predictions)
        else:
            predictions_sbox = multilabel_without_permind_predict(predictions)
        for target_byte in range(16):
            ranks_i = full_ranks(predictions_sbox[target_byte], X_attack, Metadata_attack, 0, num_traces, 10,
                                 target_byte, simulated_key)
            # We plot the results
            x_i = [ranks_i[i][0] for i in range(0, ranks_i.shape[0])]
            y_i = [ranks_i[i][1] for i in range(0, ranks_i.shape[0])]
            plt.plot(x_i, y_i, label="key_" + str(target_byte))
        plt.title('Performance of ' + model_file + ' against ' + ascad_database)
        plt.xlabel('number of traces')
        plt.ylabel('rank')
        plt.grid(True)
        plt.legend(loc='upper right')
        if (save_file != ""):
            plt.savefig(save_file)
        else:
            plt.show(block=False)

    else:
        predictions_sbox_i = predictions
        # We test the rank over traces of the Attack dataset, with a step of 10 traces
        ranks = full_ranks(predictions_sbox_i, X_attack, Metadata_attack, 0, num_traces, 10, target_byte, simulated_key)
        # We plot the results
        x = [ranks[i][0] for i in range(0, ranks.shape[0])]
        y = [ranks[i][1] for i in range(0, ranks.shape[0])]
        plt.title('Performance of ' + model_file + ' against ' + ascad_database)
        plt.xlabel('number of traces')
        plt.ylabel('rank')
        plt.grid(True)
        plt.plot(x, y)
        plt.show(block=False)
        if (save_file != ""):
            plt.savefig(save_file)
        else:
            plt.show(block=False)
