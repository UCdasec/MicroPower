#!/usr/bin/python3
import random
import sys
import numpy as np
from tools.SideChannelConstants import SideChannelConstants


def aes_internal(inp_data_byte, key_byte):
    inp_data_byte = int(inp_data_byte)
    aesSBox = SideChannelConstants.getAESSBox()
    return aesSBox[inp_data_byte ^ key_byte]


def aes_reversal(predict_byte, inp_data_byte):
    invAESSBox = SideChannelConstants.getInvAESSBox()
    predicted_key_byte = (invAESSBox[predict_byte] ^ inp_data_byte)
    return predicted_key_byte


def calc_hamming_weight(n):
    return bin(n).count("1")


def get_HW():
    HW = []
    for i in range(0, 256):
        hw_val = calc_hamming_weight(i)
        HW.append(hw_val)
    return HW


def load_whole_pack(whole_pack, attack_window):
    # hamming weight list
    HW = get_HW()

    if attack_window:
        print('[LOG] -- Using the self-defined attack window')
    else:
        print('[LOG] -- Read the attack window from the dataset package')
        attack_window = whole_pack['attack_window']
    start_idx, end_idx = attack_window[0], attack_window[1]
    try:
        traces, text_in, key = whole_pack['power_trace'], whole_pack['plain_text'], whole_pack['key']
    except:
        traces, text_in, key = whole_pack['trace_mat'], whole_pack['textin_mat'], whole_pack['key']
    # so that the byte number is from 0 to 15
    return traces, text_in, key, HW, start_idx, end_idx


def process_raw_data(whole_pack, target_byte, network_type, attack_window=0):
    # loading data and calculate its label
    traces, text_in, key, HW, start_idx, end_idx = load_whole_pack(whole_pack, attack_window)
    # print("traces shape: {}".format(traces.shape))
    key_byte = int(key[target_byte])
    labels = []
    # print(network_type)
    for i in range(text_in.shape[0]):
        text_i = text_in[i]
        label = aes_internal(text_i[target_byte], key_byte)
        if network_type == 'HW':
            label = HW[label]
            # print("here1")
        labels.append(label)

    # if hw model, make sure it is correct
    if network_type == 'HW':
        assert (9 == len(set(labels)))
    else:
        assert (256 == len(set(labels)))

    labels = np.array(labels)
    if not isinstance(traces, np.ndarray):
        traces = np.array(traces)
    traces = traces[:, start_idx:end_idx]

    inp_shape = (end_idx - start_idx, 1)
    return traces, labels, text_in, key, inp_shape


# ***************************************
def process_raw_data_shifted(whole_pack, target_byte, network_type, shifted, attack_window=0):
    # loading data and calculate its label
    traces, text_in, key, HW, start_idx, end_idx = load_whole_pack(whole_pack, attack_window)
    key_byte = int(key[target_byte])
    labels = []
    shifted_traces = []
    for i in range(text_in.shape[0]):
        text_i = text_in[i]
        label = aes_internal(text_i[target_byte], key_byte)
        if 'HW' == network_type:
            label = HW[label]
        labels.append(label)

        trace_i = traces[i]
        random_int = random.randint(0, shifted)
        trace_i = trace_i[start_idx + random_int: end_idx + random_int]
        shifted_traces.append(trace_i)

    # if hw model, make sure it is correct
    if 'HW' == network_type:
        assert (9 == len(set(labels)))
    else:
        assert (256 == len(set(labels)))

    labels = np.array(labels)
    traces = np.array(shifted_traces)

    inp_shape = (end_idx - start_idx, 1)
    return traces, labels, text_in, key, inp_shape


def sanity_check(input_layer_shape, X_profiling):
    if input_layer_shape[1] != X_profiling.shape[1]:
        print("Error: model input shape %d instead of %d is not expected ..." % (
        input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        Reshaped_X_profiling = X_profiling
    elif len(input_layer_shape) in [3,4]:
        # This is a CNN: expand the dimensions
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)
    return Reshaped_X_profiling


def binary_search(arr):
    # we need to do it in a binary search style
    low = 0
    high = arr.shape[0]
    mid = 0

    if 0 == sum(arr):
        return 0
    rtn = high
    while low <= high:
        mid = (low + high) // 2
        # if later part summation is equal to 0
        if 0 == sum(arr[mid:]):
            high = mid - 1
            rtn = mid
        # if first part summation is equal to 0
        else:
            low = mid + 1

    return rtn


def compute_min_rank(ranking_list):
    ''' try to find the last value that not convergence to 0 '''
    ranking_list = np.array(ranking_list)
    num = binary_search(ranking_list)
    return num


def test():
    test_arr = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0]
    test_arr = np.array(test_arr)
    print(binary_search(test_arr))
    print('{} {} {}'.format(test_arr[4], '\t', test_arr[5]))

    test_arr = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 5]
    test_arr = np.array(test_arr)
    print(binary_search(test_arr))
    print(test_arr.shape[0])


if __name__ == "__main__":
    test()
