# MicroPower



MicroPower is an iterative algorithm designed to automatically prune 
side channel models to their practical minimum while retaining
cross-device attack performance and quantization 
compatibility. A pre-trained parent model is processed into 
child models of decreasing size over an indefinite number of 
rounds. A routine of pruning and finetuning is applied until
reaching one of two possible stop conditions:
1. Filter sizes in the child model remain unchanged after a pruning ratio is applied. I.e. the number of filters in a layer are low enough that the calculated numbers of filters after rounding remain the same (2 filters at 0.3 pruning rate = 2 fitlers).
2. A child model fails to meet a set accuracy threshold after increasing the search space to the maximum set value. In other words, it is found that reducing the model further results in an ineffective model.

## Reference

When creating pruned models using techniques found within one should cite the following:
* ```<to be updated>```

# Content

The repository contains a full Python project using Python 
3.6 and Tensorflow 2.6.2 (exact versions should be 
unnecessary but are included for reproducibility). See the
requirement section below for further environment details. 
* ```prune_and_finetune.py``` - Performs the main pruning functionality of the iterative method.
* ```temperer.py```           - Performs the final finetuning step of the algorithm. Left out to compare model before and after. 
* ```test.py```               - Tests models on held-out testing data and generates reports.
* ```/tools/```                - Contains various accessory files used by the main files above.

## Requirements

CUDA 12.1 and CUDNN 8 are installed via Deb files from the 
NVidia Website. Instructions on Deb installation can be 
found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). 

A Conda environment is used to collect the required packages.
The ```environment.yml``` file included in the repository has the required 
packages listed and can be created new by conda via the following
command ```conda env create --name micropower --file=environment.yml```

# Usage

The algorithm requires a pretrained model to process. CNN and 
ASCADv2 architectures are supported however adding additional 
architecture support is trivial. Enable the required environment 
before running any of the following commands. 

1. Run the MicroPower pruning on a given CNN_best architecture model:
    - ```python3 prune_and_finetune.py -i /home/logan/datasets/S1_K1_200k.npz -m /home/logan/IterativePruning/inputModels/STM32/CNN2_B2_STM32Unmasked_1200to2200_ID_150e_50000t/ -o /home/logan/IterativePruning/tests/Experiment1/STM32/CNNIterativePruned_B2_STM32_1200to2200_ID_150e_50000t_Retrain_008_30per/ -lm ID -arch CNN -aw 1200_2200 -e 150 -tt 50000 -vt 10000 -tb 2 -va 0.008 -fs l2 -pt iterative```
2. Run the MicroPower pruning on a given ASCADv2 architecture model:
    - ```python3 prune_and_finetune.py -i /home/logan/datasets/ASCADv2.h5 -m /home/logan/IterativePruning/inputModels/ASCAD/ResNet_ASCADv2WithoutPermIDs_0to15000_ID_60e_420000t/ -o /home/logan/IterativePruning/tests/Experiment1/ASCAD/ResNetIterativePruned_ASCADv2WithoutPermIDs_0to15000_ID_420000t_30per_006_avg/ -lm ID -arch ASCADv2 -aw 0_15000 -e 20 -ascadV2Type withoutPermIDs -tt 420000 -vt 10000 -va 0.006 -fs l2 -pt iterative```
3. Perform the final finetuning step on a maximally pruned model:
    - ```python3 temperer.py -i /home/logan/datasets/S1_K1_200k.npz -m /home/logan/IterativePruning/tests/Experiment1/STM32/CNNIterativePruned_B2_STM32_1200to2200_ID_150e_50000t_Retrain_008_30per/Round44/ -o /home/logan/IterativePruning/tests/Experiment1/STM32/CNNIterativePruned_B2_STM32_1200to2200_ID_150e_50000t_Retrain_008_30per/TemperedRound44/ -aw 1200_2200 -lm ID -me 500 -tn 50000 -tb 2 -v```
4. Apply PTQ quantization to passed model (pruned or unpruned):
    - ```python3 prune_and_finetune.py -i /home/logan/datasets/S1_K1_200k.npz -m /home/logan/IterativePruning/tests/Experiment7/STM32/CNNIterativePruned_B2_STM32_1200to2200_ID_150e_50000t_Retrain_008_30per/TemperedRound44/ -o /home/logan/IterativePruning/tests/Experiment7/STM32/CNNIterativePruned_B2_STM32_1200to2200_ID_150e_50000t_Retrain_008_30per/TemperedRound44_INT8/ -tt 50000 -lm ID -aw 1200_2200 -tb 2 -pt ptq -qt dynamicInt8```
5. Test a model using testing traces:
    - ```python3 test.py -i /home/logan/datasets/S2_K3_100k.npz -m /home/logan/IterativePruning/tests/Experiment1/STM32/CNNIterativePruned_B2_STM32_1200to2200_ID_150e_50000t_Retrain_008_30per/TemperedRound44/ -aw 1200_2200 -lm ID -tb 2 -tn 5000```

