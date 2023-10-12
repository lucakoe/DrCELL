import os
import pickle

import pandas as pd
import scipy.io as sio
import numpy as np
import umap

umapOutDumpData = {}


def print_mat_file(file_path):
    try:
        # Load the MATLAB file
        data = sio.loadmat(file_path)

        # Print the contents of the MATLAB file
        print("MATLAB File Contents:")
        for variable_name in data:
            print(f"Variable: {variable_name}")
            print(data[variable_name])
            print("\n")
    except Exception as e:
        print(f"Error: {e}")


def getUMAPOut(data, dumpPath, n_neighbors=20,
               min_dist=0.0,
               n_components=2,
               random_state=42, ):
    paramKey = (n_neighbors, min_dist, n_components, random_state)
    umapOutDfDump = {}

    # Checks the global variable if the dumpfile with this path was already called. If so instead of loading it for every call of the function it takes the data from there
    if not (dumpPath in umapOutDumpData):
        # Check if the file exists
        if os.path.exists(dumpPath):
            with open(dumpPath, 'rb') as file:
                umapOutDumpData[dumpPath] = pickle.load(file)
        else:
            # If the file doesn't exist, create it and write something to it
            with open(dumpPath, 'wb') as file:
                pickle.dump(umapOutDfDump, file)
                umapOutDumpData[dumpPath] = umapOutDfDump

            print(f"The file '{dumpPath}' has been created.")

    umapOutDfDump = umapOutDumpData[dumpPath]

    if not(paramKey in umapOutDfDump):
        print(
            f"Generate UMAP: File = {os.path.basename(dumpPath)}, n_neighbors = {n_neighbors}, min_dist = {min_dist}")
        umapObject = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
        )
        umapOutDfDump[paramKey] = umapObject.fit_transform(data)

        with open(dumpPath, 'wb') as file:
            umapOutDumpData[dumpPath] = umapOutDfDump
            pickle.dump(umapOutDfDump, file)
    print(
        f"Return UMAP: File = {os.path.basename(dumpPath)}, n_neighbors = {n_neighbors}, min_dist = {min_dist}")
    return umapOutDfDump[paramKey]


def loadAndPreprocessData(cFile):
    # load data
    print(cFile)
    g = sio.loadmat(cFile)
    # g = h5py.File(cFile,'r')
    # X = np.array(g['X'])
    X = np.array(g['UMAPmatrix'])
    Y = np.array(g['UMAPmatrix'])
    # Y = np.array(g['Y'])
    # Y = np.transpose(Y)
    matrixLegend = np.array(g['matrixLegendArray'])
    # groups = np.array(g['redIdxAll'])
    # groups = np.reshape(groups,X.shape[0])
    # groups = groups.astype(int)

    # groups = np.array(g['groupIdx'])

    # groups = np.array(g['matrixLegend'])
    # for iNeuron in range(0,len(groups)):
    #     if groups[iNeuron,4] == 'Innate':
    #         groups[iNeuron,4] = 1
    #     elif groups[iNeuron,4] == 'Audio Task Without Delay':
    #         groups[iNeuron,4] = 2
    #     elif groups[iNeuron,4] == 'Audio Task with delay':
    #         groups[iNeuron,4] = 3
    #     elif groups[iNeuron,4] == 'Innate2':
    #         groups[iNeuron,4] = 4
    # groups = np.reshape(groups[:,4],X.shape[0])
    # groups = groups.astype(int)

    # g.close

    # Extract the 'UMAPmatrix' array from the loaded data and create a Pandas DataFrame from 'UMAPmatrix'
    umapDf = pd.DataFrame(g['UMAPmatrix'])

    # Extract the 'matrixLegendArray' array from the loaded data and create a Pandas DataFrame from 'matrixLegendArray'
    matrixLegendDf = pd.DataFrame(g['matrixLegendArray'],
                                  columns=["Animal", "Recording", "Neuron", "NbFrames", "Task", "RedNeurons",
                                           "ChoiceAUCs", "IsChoiceSelect", "StimAUCs", "IsStimSelect"])

    # Convert the 'Task' column to integers
    matrixLegendDf['Task'] = matrixLegendDf['Task'].astype(int)
    # Convert Float to Boolean
    matrixLegendDf["IsStimSelect"] = matrixLegendDf["IsStimSelect"].apply(lambda x: x >= 1.0)
    matrixLegendDf["IsChoiceSelect"] = matrixLegendDf["IsChoiceSelect"].apply(lambda x: x >= 1.0)

    useIdx = np.invert(np.isnan(np.sum(Y, axis=0)))
    Y = Y[:, useIdx]

    # run umap twice, once for spatial and once for temporal clusters
    # umapOut = umap.UMAP(
    #     n_neighbors=30,
    #     min_dist=0.0,
    #     n_components=2,
    #     random_state=42,
    #     ).fit_transform(X)
    # print('Umap vals: ' + str(umapOut.shape))

    # show results
    # cIdx = np.random.permutation(groups.size)
    # fig1, ax1 = plt.subplots()
    # cColor = sns.color_palette("Dark2", np.max([groups])+1);
    # ax1.scatter(umapOut[cIdx, 0], umapOut[cIdx, 1], c=[cColor[x] for x in groups[cIdx]], s=2)
    # #plt.axis('square')
    # ax1.set_aspect('equal', adjustable = 'datalim')
    # plt.show()

    # temporal clusters
    return umapDf, matrixLegendDf, Y