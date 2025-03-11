# genInfoPlots.py - Start 10/30/23 - Logan Reichling
# genInfoPlots.py is a script that takes a numpy array and plots the key rank curve, adding some helpful information
import json
import os.path
import matplotlib.pyplot as plt
import numpy as np


def load_npz_file(file_name):
    """
    load_npz_file takes a file name and loads the numpy array from the file.
    :param file_name: The name of the file to load the numpy array from.
    :return: The numpy array loaded from the file.
    """
    with np.load(file_name) as dataArray:
        return dataArray['y']


def confirmShape(array):
    """
    get_shape takes a numpy array and returns the shape of the array.
    :param array: The numpy array to get the shape of.
    :return: The shape of the numpy array.
    """
    shapeNumpy = array.shape
    assert len(shapeNumpy) == 1, "Array is not one-dimensional"
    return shapeNumpy


def findZeroIndices(array):
    """
    find_zero_indices takes a numpy array and returns the indices of the first and last zero values.
    :param array: The numpy array to find the indices of the first and last zero values.
    :return: The indices of the first and last zero values as a tuple
    """
    KEY_RANK_ZERO_THRESHOLD = 0.5

    # BOUNCE_SIZE = 2
    firstBecomeZero = -1
    final_become_zero = -1
    for i in range(len(array)):
        if array[i] <= KEY_RANK_ZERO_THRESHOLD and firstBecomeZero == -1:
            firstBecomeZero = i
        # elif array[i] <= KEY_RANK_ZERO_THRESHOLD and firstBecomeZero != -1:
        #    final_become_zero = i
        #    break
    return firstBecomeZero, final_become_zero


def create_cropped_array(array, becomeZero):
    """
    create_cropped_array takes a numpy array and returns a cropped version of the array.
    :param array: The numpy array to crop.
    :param becomeZero: The index of the last zero value in the array.
    :return: The cropped numpy array, to better visualize the curve.
    """
    return array[:becomeZero + 20]


def plotDefaultRankCurve(array, byteStrNum, leakageModel, show=False):
    plt.figure(figsize=(8, 6))
    plt.plot(array, color='red')
    plt.title('Leakage model: {}, target byte: {}'.format(leakageModel, byteStrNum))
    plt.xlabel('Number of trace')
    plt.ylabel('Key Rank')
    if show:
        plt.show()
        plt.close()


def plotKeyRankCurve(array, byteStrNum, xLabel, yLabel, device, leakageModel, firstBecomeZero, lastBecomeZero,
                     show=False):
    """
    plot_key_rank_curve takes a numpy array and plots the key rank curve.
    :param leakageModel:
    :param show:
    :param yLabel:
    :param device:
    :param byteStrNum:
    :param xLabel:
    :param array: The numpy array to plot the key rank curve of.
    :param firstBecomeZero: The index of the first zero value in the array.
    :param lastBecomeZero: The index of the last zero value in the array.
    :return: None
    """
    plt.plot(array, label="Key Rank Curve")
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(f"{device} Key Rank Curve for Byte {byteStrNum} Target.\nLeakage Model: {leakageModel}")
    if firstBecomeZero != -1:
        plt.axvline(x=firstBecomeZero, color='g', linestyle="--", label=f"First zero at {firstBecomeZero}")
    if lastBecomeZero != -1:
        plt.axvline(x=lastBecomeZero, color='r', linestyle="--", label=f"Last zero at {lastBecomeZero}")
    plt.legend()
    if show:
        plt.show()


def createTimeGraph(array, xLabel, yLabel, roundNum, datasetName, leakageModel, exportDir, saveName, show=False):
    """
    plotRoundTimes takes and plots an array representing the time per pruning round.
    :param array:           Array of time per round data
    :param xLabel:          Label for x-axis, i.e. 'Round'
    :param yLabel:          Label for y-axis, i.e. 'Time (s)'
    :param roundNum:        Number of rounds
    :param datasetName:     Name of the dataset
    :param leakageModel:    Leakage model utilized
    :param exportDir:       Directory to save within
    :param saveName:        Time graph save name
    :param show:            Boolean to trigger the graph to pop-up; not for headless systems
    :return: None
    """
    fig, ax1 = plt.subplots()
    ax1.tick_params(length=6, width=3)
    #fig.tight_layout()
    plt.bar(x=range(len(array)), height=array, label="Time per round", color='#11AA11')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel, color='#117711')
    # plt.title(f"Pruning time over {roundNum} rounds for {datasetName} (0.7 initial rate)")
    # for i in range(len(array)):
    #     plt.text(i, array[i], f"   {array[i]:.1f} s", ha='center', va='baseline', rotation=90)
    plt.text(6, max(array) + 68080 * 0.9, f"Total time: {int(sum(array)+0.5):,} s", ha='left', va='top', rotation=0,
             bbox=dict(facecolor='none', edgecolor='#dddddd', boxstyle='round'), fontsize="small")
    # Make the y-axis logarithmic
    plt.yscale('log')
    plt.yticks([100, 1000, 10000, 100000], color='#117711', size=20)
    x_ticks = np.arange(0, len(array)+1, 8)
    plt.xticks(x_ticks, [str(i+1) for i in x_ticks])

    ax2 = ax1.twinx()
    # 0.1 initial parameter counts
    # ax2.plot(, color='blue', lw=2)

    # 0.7 initial parameter counts
    paraCounts0_7 = [5098862,1356301,1110956,912196,751845,616255,508979,418584,346511,286212,238323,198274,163797,
              138624,117155,98210,82339,68512,59055,50822,43874,37924,32364,27678,23779,20154,18024,16356,14720,
              13116,11934,10770,9624,8870,8124,7386,6656,5934,5576,5220,4866,4514,4164,3816,3470,3126]
    paraCounts0_1 = [43896874,35644347,28972815,23561429,19131008,15562013,12665406,10291139,8380419,6820146,
                5554101,4532919,3706542,3029231,2470989,2016491,1656841,1354431,1109251,911488,751173,615631,508389,
                423938,351309,290506,243342,202801,168867,143065,121200,101876,85660]
    ax2.plot(paraCounts0_1, color='blue', lw=2)

    ax2.set_yscale('log')
    filled_marker_style = dict(marker='o', markersize=15,
                               color='blue', linestyle='',
                               markerfacecolor='tab:blue',
                               markerfacecoloralt='lightsteelblue',
                               markeredgecolor='blue')
    ax2.plot(range(0, len(paraCounts0_1), 3) ,paraCounts0_1[::3], fillstyle="none", **filled_marker_style)
    ax2.set_yticks([1000, 10000, 100000, 1000000, 10000000, 100000000])
    #ax2.set_yticklabels(['0', '1M', '2M', '3M', '4M', '5M'])
    ax2.set_ylabel('No. of Parameters', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue', length=6, width=3)

    plt.margins(x=0.01)
    ax1.set_xmargin(0.01)
    ax2.set_xmargin(0.01)
    # plt.yticks(y_ticks)


    plt.subplots_adjust(left=0.18, right=0.82, top=0.95, bottom=0.2)
    # plt.legend()

    plt.savefig(os.path.join(exportDir, saveName + '.png'))
    plt.savefig(os.path.join(exportDir, saveName + '.pdf'))

    if show:
        plt.show()



# Can run as console application or run directly from IDE
# Console application: python genInfoPlots.py <path_to_npz_file.npz> <path_to_save_figure.png>
if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (6, 4)
    plt.rcParams.update({'font.size': 22})
    # Data for 0.7 initial rate
    # test = np.array(json.loads("[19369.019325494766, 41890.94952130318, 57057.55313563347, 268.5455334186554, 284.9875648021698, 263.67587423324585, 240.90214157104492, 243.25974464416504, 239.144549369812, 240.3862545490265, 238.168226480484, 244.81861901283264, 240.05222868919373, 233.86265182495117, 221.41871213912964, 233.0430166721344, 226.3417453765869, 225.01850008964539, 231.01800918579102, 222.95390677452087, 223.99113249778748, 220.9342486858368, 220.64256620407104, 213.94870495796204, 211.35581946372986, 208.67135667800903, 210.71126627922058, 215.23349285125732, 209.54111695289612, 215.781818151474, 217.0392725467682, 203.11096668243408, 218.65720129013062, 214.01153564453125, 200.90342903137207, 218.42123532295227, 209.8483190536499, 221.1427710056305, 218.43140959739685, 210.2423779964447, 201.62474656105042, 217.38595747947693, 202.39482522010803, 218.8649218082428, 212.11258697509766, 219.43437600135803]"))
    # Data for 0.1 initial rate
    test = np.array(json.loads("[1575.5122170448303, 1279.4453971385956, 1068.8557889461517, 915.338404417038, 770.3907480239868, 665.3246924877167, 577.9745376110077, 515.0907742977142, 463.3342981338501, 412.1751585006714, 377.7999269962311, 361.9987885951996, 349.6514000892639, 312.4356942176819, 308.54790353775024, 292.29234981536865, 280.5480499267578, 280.51874709129333, 271.2724792957306, 257.14640378952026, 264.56484389305115, 265.53524565696716, 254.98613142967224, 254.86340641975403, 230.930739402771, 243.3905885219574, 247.62684106826782, 239.37980914115906, 243.5864977836609, 236.48883652687073, 233.55223298072815, 4712.166446208954, 13054.116265773773]"))

    createTimeGraph(test, "No. of Rounds", "Time (s)", len(test), "S1_K1_150K_EM", "ID",
                    "C:\\Users\\Logan Reichling\\PycharmProjects\\IterativePruning\\misc", "test", True)

    # if not sys.stdout.isatty():
    #     # Load single numpy .npz file
    #     data = load_npz_file(
    #         "testDir/CNN_B1_XMegaUnmasked_1800to2800_HW_test_10000t_X1K1200k/rank_dir/ranking_raw_data.npz")
    #
    #     # Save the shape of the numpy array to a variable and confirm that it is a one-dimensional array
    #     shape = confirmShape(data)
    #     first_become_zero, last_become_zero = findZeroIndices(data)  # Find important indices
    #     if last_become_zero != -1:
    #         cropped_data = create_cropped_array(data, last_become_zero)  # Crop original dataset
    #     elif first_become_zero != -1:
    #         cropped_data = create_cropped_array(data, first_become_zero)
    #     else:
    #         cropped_data = data
    #     plotKeyRankCurve(cropped_data, "TEST", "Number of Traces", "Key Rank", "TEST", "TEST", first_become_zero,
    #                      last_become_zero, show=True)  # Plot the rank curve
    #     plt.savefig("key_rank_curve_test_cropped.png")
    #
    # else:
    #     passedArgs = sys.argv
    #     if len(passedArgs) == 2:
    #         if os.path.exists(passedArgs[1]) and os.path.isfile(passedArgs[1]):
    #             data = load_npz_file(passedArgs[1])
    #             shape = confirmShape(data)
    #             first_become_zero, last_become_zero = findZeroIndices(data)
    #             if last_become_zero != -1:
    #                 cropped_data = create_cropped_array(data, last_become_zero)  # Crop original dataset
    #             elif first_become_zero != -1:
    #                 cropped_data = create_cropped_array(data, first_become_zero)
    #             else:
    #                 cropped_data = data
    #             # Plot the rank curve
    #             plotKeyRankCurve(cropped_data, "TEST", "Number of Traces", "Key Rank", "TEST", "TEST",
    #                              first_become_zero, last_become_zero, show=True)
    #             plt.savefig("key_rank_curve_test_cropped.png")
    #
    #         # Batch processing for all tests, should be the preferred method
    #         # If passed a dir, process all directories that have correct format
    #         # Ex: command "genInfoPlots.py /SoftPower-master/", will process in all like:
    #         # "/SoftPower-master/CNN_B0_XMegaUnmasked_1800to2800_HW_test_10000t_X1K1100k/rank_dir/ranking_raw_data.npz"
    #
    #         elif os.path.exists(passedArgs[1]) and os.path.isdir(passedArgs[1]):
    #             resultFigPaths = list()
    #             resultFigDescriptions = list()
    #             dirName = os.path.basename(os.path.normpath(passedArgs[1]))
    #             namingMatch = re.compile(
    #                 r"^CNN(.*?)_B(\d{1,2})_(.+?)_(\d{1,5})to(\d{3,5})_(HW|ID)_test_(\d{3,5})t_(.+)")
    #             for fileOrFolder in os.scandir(passedArgs[1]):
    #                 if fileOrFolder.is_dir():
    #                     # Check if directory matches standard directory name
    #                     directoryMatches = namingMatch.match(fileOrFolder.name)
    #                     if directoryMatches is not None:
    #                         matchGroups = directoryMatches.groups()
    #                         # Check if directory has a ranking_raw_data.npz file
    #                         if os.path.exists(fileOrFolder.path + "/rank_dir/ranking_raw_data.npz"):
    #                             data = load_npz_file(os.path.join(fileOrFolder.path + "/rank_dir/ranking_raw_data.npz"))
    #                             shape = confirmShape(data)
    #                             first_become_zero, last_become_zero = findZeroIndices(data)
    #                             if last_become_zero != -1:
    #                                 cropped_data = create_cropped_array(data, last_become_zero)
    #                             elif first_become_zero != -1:
    #                                 cropped_data = create_cropped_array(data, first_become_zero)
    #                             else:
    #                                 cropped_data = data
    #
    #                             print(f"First 10 results from {fileOrFolder.name}")
    #                             print(cropped_data[:10])
    #
    #                             # Generate graph with filled-in info
    #                             plotKeyRankCurve(cropped_data, matchGroups[0], "Number of Traces", "Key Rank",
    #                                              matchGroups[1], matchGroups[4], first_become_zero, last_become_zero)
    #                             plt.savefig(os.path.join(fileOrFolder.path +
    #                                                      f"/rank_dir/key_rank_curve_B{matchGroups[0]}_cropped.png"))
    #                             resultFigPaths.append(os.path.join(fileOrFolder.path +
    #                                                                f"/rank_dir/key_rank_curve_B{matchGroups[0]}_cropped.png"))
    #                             resultFigDescriptions.append(f"Dataset: {matchGroups[6]}\nByte: {matchGroups[0]}")
    #                             plt.clf()
    #
    #                         else:
    #                             print(f"WARNING: Result folder \"{fileOrFolder.path}\" missing npz data!")
    #                     else:
    #                         print(f"WARNING: No naming matches on passed directory!")
    #             # Generate final report if numpy data was processed
    #             if len(resultFigPaths) > 0:
    #                 pass
    #
    #         else:
    #             print("File/directory does not exist.")
    #         print("Done!")
    #
    #     elif len(passedArgs) == 3:
    #         data = load_npz_file(passedArgs[1])
    #         shape = confirmShape(data)
    #         first_become_zero, last_become_zero = findZeroIndices(data)
    #         if last_become_zero != -1:
    #             cropped_data = create_cropped_array(data, last_become_zero)  # Crop original dataset
    #         elif first_become_zero != -1:
    #             cropped_data = create_cropped_array(data, first_become_zero)
    #         else:
    #             cropped_data = data
    #         # Plot the rank curve
    #         plotKeyRankCurve(cropped_data, "TEST", "Number of Traces", "Key Rank", "TEST", "TEST", first_become_zero,
    #                          last_become_zero, show=True)
    #         plt.savefig(passedArgs[2])
