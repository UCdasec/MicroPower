import random
import sys
from collections import defaultdict
from tools.SideChannelConstants import SideChannelConstants
import numpy as np
from sklearn import preprocessing


def preprocess_data(x_data, method):
    if method == 'norm':     # 'horizontal_standardization':
        print('[LOG] -- Using {} method to preprocessing the data.'.format(method))
        mn = np.repeat(np.mean(x_data, axis=1, keepdims=True), x_data.shape[1], axis=1)
        std = np.repeat(np.std(x_data, axis=1, keepdims=True), x_data.shape[1], axis=1)
        x_data = (x_data - mn)/std
    elif method == 'scaling':    #  'horizontal_scaling':
        print('[LOG] -- Using {} method to preprocessing the data.'.format(method))
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x_data.T)
        x_data = scaler.transform(x_data.T).T
    else:
        print('[LOG] -- No preprocessing applied to the data.')
    return x_data


def calc_hamming_weight(n):
    return bin(n).count("1")


def get_HW():
    HW = []
    for i in range(0, 256):
        hw_val = calc_hamming_weight(i)
        HW.append(hw_val)
    return HW


HW = get_HW()


def aes_internal(inp_data_byte, key_byte):
    inp_data_byte = int(inp_data_byte)
    aesSBox = SideChannelConstants.getAESSBox()
    return aesSBox[inp_data_byte ^ key_byte]


def create_hw_label_mapping():
    ''' this function return a mapping that maps hw label to number per class '''
    HW = defaultdict(list)
    for i in range(0, 256):
        hw_val = calc_hamming_weight(i)
        HW[hw_val].append(i)
    return HW


def get_one_label(text_i, target_byte, key_byte, leakage_model):
    ''''''
    label = aes_internal(text_i[target_byte], key_byte)
    if 'HW' == leakage_model:
        label = HW[label]
    return label


def get_labels(plain_text, key_byte, target_byte, leakage_model):
    ''' get labels for a batch of data '''
    labels = []
    for i in range(plain_text.shape[0]):
        text_i = plain_text[i]
        label = get_one_label(text_i, target_byte, key_byte, leakage_model)
        labels.append(label)

    if leakage_model == 'HW':
        try:
            assert(set(labels) == set(list(range(9))))
        except Exception:
            print('[LOG] -- Not all class have data: ', set(labels))
    else:
        try:
            assert(set(labels) == set(range(256)))
        except Exception:
            print('[LOG] -- Not all class have data: ', set(labels))
    labels = np.array(labels)
    return labels


def shift_the_data(shifted, attack_window, trace_mat, textin_mat):
    start_idx, end_idx = attack_window[0], attack_window[1]

    if shifted:
        print('[LOG] -- Data will be shifted in range: ', [0, shifted])
        shifted_traces = []
        for i in range(textin_mat.shape[0]):
            random_int = random.randint(0, shifted)
            trace_i = trace_mat[i, start_idx+random_int:end_idx+random_int]
            shifted_traces.append(trace_i)
        trace_mat = np.array(shifted_traces)
    else:
        print('[LOG] -- No random delay apply to the data')
        trace_mat = trace_mat[:, start_idx:end_idx]

    return trace_mat, textin_mat


def downsampling(traces):
    ori_dim = traces.shape[1]
    expected_dim = 1000
    gcd_val = np.gcd(ori_dim, expected_dim)

    ori_step = ori_dim // gcd_val
    exp_step = expected_dim // gcd_val

    diff = abs(exp_step-ori_step)
    del_val = []
    for i in range(0, ori_dim+ori_step, ori_step):
        for j in range(1, diff+1):
            tmp = i - j
            del_val.append(tmp)

    traces = np.delete(traces, del_val, axis=1)
    return traces


def unpack_data(whole_pack):
    try:
        traces, plain_text, key = whole_pack['power_trace'], whole_pack['plain_text'], whole_pack['key']
    except KeyError:
        try:
            traces, plain_text, key = whole_pack['power_trace'], whole_pack['plaintext'], whole_pack['key']
        except KeyError:
            traces, plain_text, key = whole_pack['trace_mat'], whole_pack['textin_mat'], whole_pack['key']
    return traces, plain_text, key


def load_data_base(whole_pack, attack_window, method, trace_num=0, shifted=0):
    if isinstance(attack_window, str):
        tmp = attack_window.split('_')
        attack_window = [int(tmp[0]), int(tmp[1])]

    traces, plain_text, key = unpack_data(whole_pack)

    if trace_num:
        traces = traces[:trace_num, :]
        plain_text = plain_text[:trace_num, :]

    traces, plain_text = shift_the_data(shifted, attack_window, traces, plain_text)

    # if traces.shape[1] > 1000:
    #     ori_dim = traces.shape[1]
    #     traces = downsampling(traces)
    #     new_dim = traces.shape[1]
    #     assert(new_dim == 1000)
    #     print('sample dim are down from {} to {}'.format(ori_dim, new_dim))

    if method:
        traces = preprocess_data(traces, method)
    return traces, plain_text, key


def load_data_base_test(whole_pack, attack_window, method, trace_num=0, shifted=0):
    if isinstance(attack_window, str):
        tmp = attack_window.split('_')
        attack_window = [int(tmp[0]), int(tmp[1])]

    traces, plain_text, key = unpack_data(whole_pack)

    if trace_num:
        traces = traces[-trace_num:, :]  # Get from back
        plain_text = plain_text[-trace_num:, :]

    traces, plain_text = shift_the_data(shifted, attack_window, traces, plain_text)

    # if traces.shape[1] > 1000:
    #     ori_dim = traces.shape[1]
    #     traces = downsampling(traces)
    #     new_dim = traces.shape[1]
    #     assert(new_dim == 1000)
    #     print('sample dim are down from {} to {}'.format(ori_dim, new_dim))

    if method:
        traces = preprocess_data(traces, method)
    return traces, plain_text, key


def sanity_check(input_layer_shape, X_profiling):
    if input_layer_shape[1] != X_profiling.shape[1]:
        print("Error: model input shape %d instead of %d is not expected ..." % (
        input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        Reshaped_X_profiling = X_profiling
    elif len(input_layer_shape) == 3:
        # This is a 1D CNN: expand the dimensions
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    elif len(input_layer_shape) == 4:
        # This is a 2D CNN: expand the dimensions
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1, 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)
    return Reshaped_X_profiling


if __name__ == "__main__":
    test_arr = np.zeros((100, 1300), dtype=float)
    new_arr = downsampling(test_arr)
    assert(new_arr.shape[1] == 1000)
    print('array dimension change from {} to {}'.format(test_arr.shape[1], new_arr.shape[1]))

