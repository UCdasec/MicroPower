# FPGM_L2_Score.py - Refactor Start 12/11/23 - Logan Reichling
# This file is a refactor of the original 'Rank_generation.py' in order to be more focused.
# Original Authors: Mabon Ninan and Haipeng Li

# Imports
import os
import time
import argparse                      # Import argument parsing library
from scipy.spatial import distance   # Import distance calculation from SciPy
import numpy
import pandas
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Conv1D, Add, Dense


# Define a function for quantization and mapping of weights
def mapping(W, min_w, max_w):
    scale_w = (max_w - min_w) / 100
    min_arr = numpy.full(W.shape, min_w)
    q_w = numpy.round((W - min_arr) / scale_w).astype(numpy.uint8)
    return q_w


# Define a function to extract weights from a model
def l2_scores(model, outputDir, verbose):
    Results = list()
    idx_results = list()
    r = list()
    warningFlag = False
    for layer in model.layers:
        if type(layer) is Conv2D:
            warningFlag = True
    if warningFlag:
        print("WARNING: L2 Score generation tested for 2D models that have only redundant 2nd dimension!")
    for layer in model.layers:
        if type(layer) in [Conv2D, Conv1D, Dense]:  #
            print(layer.name) if verbose is True else None
            a = layer.get_weights()[0]
            n_filters = a.shape[-1]
            for i in range(n_filters):
                w = a[..., i] if type(layer) != Conv2D else a[..., i][:, :, 0]  # 2D hack
                r.append(numpy.linalg.norm(w, 2))
            r = mapping(numpy.array(r), numpy.min(r), numpy.max(r))
            Results.append(sorted(r, reverse=True))
            idx_dis = numpy.argsort(r, axis=0)
            idx_results.append(idx_dis)
            r = list()
    os.makedirs(outputDir, exist_ok=True)
    df = pandas.DataFrame(Results, index=None)
    df.to_csv(os.path.join(outputDir, "l2.csv"), header=False)
    df = pandas.DataFrame(idx_results, index=None)
    df.to_csv(os.path.join(outputDir, "l2_idx.csv"), header=False)


# Define a function to apply FPGM (Filter Pruning via Geometric Median) on a model
def fpgm_scores(model, outputDir, verbose, dist_type="l2"):
    results = list()
    idx_results = list()
    r = list()
    for layer in model.layers:
        if "conv" in layer.name:  # or "fc" in layer.name
            print(layer.name) if verbose is True else None
            w = layer.get_weights()[0]
            weight_vec = numpy.reshape(w, (-1, w.shape[-1]))
            if dist_type == "l2" or "l1":
                dist_matrix = distance.cdist(numpy.transpose(weight_vec), numpy.transpose(weight_vec), 'euclidean')
            elif dist_type == "cos":
                dist_matrix = 1 - distance.cdist(numpy.transpose(weight_vec), numpy.transpose(weight_vec), 'cosine')
            squeeze_matrix = numpy.sum(numpy.abs(dist_matrix), axis=0)
            distance_sum = sorted(squeeze_matrix, reverse=True)
            idx_dis = numpy.argsort(squeeze_matrix, axis=0)
            r = mapping(numpy.array(distance_sum), numpy.min(distance_sum), numpy.max(distance_sum))
            results.append(r)
            idx_results.append(idx_dis)
            r = list()
    os.makedirs(outputDir, exist_ok=True)
    df = pandas.DataFrame(results, index=None)
    df.to_csv(os.path.join(outputDir, "fpgm.csv"), header=False)
    df = pandas.DataFrame(idx_results, index=None)
    df.to_csv(os.path.join(outputDir, "fpgm_idx.csv"), header=False)


# Parse command line arguments
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',  '--input_model_dir', help='Input model to calculate l2 or FPGM scores')
    parser.add_argument('-o',  '--output_dir', help='Output directory for l2 or FPGM scores')
    parser.add_argument('-fs', '--filter_score', choices={'l2', 'fpgm'}, help='Choice of model score algorithm')
    parserOptions = parser.parse_args()
    return parserOptions


# if __name__ == '__main__':
#     opts = parseArgs()
#     loadedModel = load_model(os.path.join(opts.input_model_dir, 'model', 'best_model.h5'))
#     loadedModel.summary()
#     if opts.filter_score == 'l2':
#         start = time.perf_counter()
#         l2_scores(loadedModel, opts.output_dir, True)
#         end = time.perf_counter()
#         print(f"Time to calculate L2 scores (seconds): {(end - start)} seconds")
#     elif opts.filter_score == 'fpgm':  # == 'fpgm'
#         start = time.perf_counter()
#         fpgm_scores(loadedModel, opts.output_dir, True)
#         end = time.perf_counter()
#         print(f"Time to calculate FPGM scores (seconds): {(end - start)} seconds")

if __name__ == "__main__":
    testModel = load_model(r"C:\Users\Logan Reichling\Desktop\ResNet_ASCADv2WithoutPermIDs_0to15000_ID_60e_420000t\model\best_model.h5")
    outDir = r"C:\Users\Logan Reichling\Desktop\ResNet_ASCADv2WithoutPermIDs_0to15000_ID_60e_420000t"
    l2_scores(testModel, outDir, True)

