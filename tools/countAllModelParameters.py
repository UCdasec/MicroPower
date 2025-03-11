import gc
import os
import fnmatch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.platform import build_info
from tensorflow.keras.models import load_model
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def find_h5_files(directory):
    h5_files = []
    for root, _, files in os.walk(directory):
        for filename in fnmatch.filter(files, '*.h5'):
            h5_files.append(os.path.join(root, filename))
    return h5_files


def returnParameters(kerasH5File):
    model = load_model(kerasH5File)
    parameters = model.count_params()
    del model
    gc.collect()
    return parameters


if __name__ == "__main__":
    # Take the path as passed argument
    import sys
    if len(sys.argv) != 2:
        print("Usage: python countAllModelParameters.py <directory_to_search>")
        sys.exit(1)
    directory_to_search = sys.argv[1]
    if os.path.isdir(directory_to_search):
        h5_files = find_h5_files(directory_to_search)
        print(f"Found {len(h5_files)} .h5 files:")
        for file_path in h5_files:
            print(file_path, returnParameters(file_path))
    else:
        print("The provided path is not a directory.")