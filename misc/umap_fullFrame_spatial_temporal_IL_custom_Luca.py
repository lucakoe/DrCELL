# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:42:38 2020

@author: Simon
"""
import pickle

import numpy as np
import pandas
# import h5py
import umap
# import hdbscan
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotting
import util

# skips UMAP creation if false and takes dump
createUMAP = True
# skips Spike Plot generation if false and takes dump
generatePlots = False
generateDiagnosticPlots = True
plotImagesFolder = r'/C:/Users/koenig/Documents/GitHub/twoP/Playground/Luca/PlaygoundProject/data/temp/plot_images'

allFiles = [r'../data/Umap_2530_2532_Array.mat']
outputPath = r'../data'
# ['C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate2530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixTaskWithoutDelay2530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixTaskWithDelay2530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate22530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate2532.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixAudioTaskWithoutDelay2532.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixAudioTaskWithDelay2532.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate22532.mat']
# 
# allFiles = ['Q:\\BpodImager\\umapClust\\20210309T2356_fullA.mat'] #first file


for cFile in allFiles:
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
    umap_df = pd.DataFrame(g['UMAPmatrix'])

    # Extract the 'matrixLegendArray' array from the loaded data and create a Pandas DataFrame from 'matrixLegendArray'
    matrix_legend_df = pd.DataFrame(g['matrixLegendArray'],
                                    columns=["Animal", "Recording", "Neuron", "NbFrames", "Task", "RedNeurons",
                                             "ChoiceAUCs", "IsChoiceSelect", "StimAUCs", "IsStimSelect"])

    # Convert the 'Task' column to integers
    matrix_legend_df['Task'] = matrix_legend_df['Task'].astype(int)

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

    # Convert the Matplotlib plots to base64-encoded PNGs

    if (generatePlots):
        for i in range(umap_df.shape[0]):
            plotting.plotAndSaveSpikes(i, umap_df, plotImagesFolder)

    tempUmapOut = pandas.DataFrame
    umapObject = umap.UMAP
    if (createUMAP):
        umapObject = umap.UMAP(
            n_neighbors=20,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        )
        tempUmapOut = umapObject.fit_transform(Y)
        print('Umap vals: ' + str(tempUmapOut.shape))

        # for debugging, dumps data to save time
        with open(r"../data/temp/tempUmapOut.pkl", 'wb') as file:
            pickle.dump(tempUmapOut, file)
        with open(r"../data/temp/umapObject.pkl", 'wb') as file:
            pickle.dump(umapObject, file)

    with open(r"../data/temp/tempUmapOut.pkl", 'rb') as file:
        tempUmapOut = pickle.load(file)
    with open(r"../data/temp/umapObject.pkl", 'rb') as file:
        umapObject = pickle.load(file)

    cell_df = pd.DataFrame(tempUmapOut, columns=['x', 'y'])
    cell_df.index = range(len(cell_df))
    matrix_legend_df.index = range(len(matrix_legend_df))
    cell_df = cell_df.merge(matrix_legend_df, left_index=True, right_index=True)
    cell_df['Task'] = cell_df['Task'].astype(str)

    # plots UMAP via Bokeh
    plotting.plotBokeh(cell_df, Y)

    # show results
    cIdx = np.random.permutation(matrix_legend_df["Task"].size)
    fig1, ax1 = plt.subplots()
    # cColor = sns.color_palette("Dark2", np.max([groups])+1);
    cColor = sns.color_palette("Spectral", np.max([matrix_legend_df["Task"].astype(float).astype(int)]) + 1);
    ax1.scatter(tempUmapOut[cIdx, 0], tempUmapOut[cIdx, 1], c=[cColor[x] for x in matrix_legend_df["Task"][cIdx]], s=2)
    # plt.axis('square')
    ax1.set_aspect('equal', adjustable='datalim')
    plt.show()

    # sio.savemat(cFile.replace('.mat','_umap_allAreas.mat'),
    #                 {'umapOut':umapOut,
    #                  'tempUmapOut':tempUmapOut,
    #                  })

    # Save UMAP coordinates and labels to see in Matlab
    umap_data = np.column_stack((tempUmapOut, matrix_legend_df["Task"]))
    # file_name = os.path.basename(cFile)
    # sio.savemat(r'\\Naskampa\lts\2P_PuffyPenguin\2P_PuffyPenguin\Umap_Analysis\output_' + file_name, {'umap_data': umap_data})

    if generateDiagnosticPlots:
        (plotting.generateDiagnosticPlots(umapObject, Y))
