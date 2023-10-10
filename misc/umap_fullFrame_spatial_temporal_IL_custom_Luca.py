# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:42:38 2020

@author: Simon
"""
import os
import sys

new_directory = r"C:\Users\koenig\Documents\GitHub\twoP\Playground\Luca\PlaygoundProject"
os.chdir(os.path.join(new_directory, "misc"))
sys.path.append(new_directory)

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
createNewUMAP = False
# skips Spike Plot generation if false and takes dump
generateAndSaveSpikePlots = False
generateDiagnosticPlots = False
generateUMAPParameters = False
spikePlotImagesPath = r'/C:/Users/koenig/Documents/GitHub/twoP/Playground/Luca/PlaygoundProject/data/temp/plot_images'

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

    if (generateAndSaveSpikePlots):
        for i in range(umap_df.shape[0]):
            plotting.plotAndSaveSpikes(i, umap_df, spikePlotImagesPath)

    tempUmapOut = pandas.DataFrame
    umapObject = umap.UMAP

    if generateUMAPParameters:
        n_neighbors_values = range(5, 51, 1)
        min_dist_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
                           0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31,
                           0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46,
                           0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62,
                           0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77,
                           0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93,
                           0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
        n_components_values = [2]
        for n_neighbors_value in n_neighbors_values:
            for min_dist_value in min_dist_values:
                for n_components_value in n_components_values:
                    print(
                        f"Current UMAP: n_neighbors = {n_neighbors_value}, min_dist = {min_dist_value}, n_components = {n_components_value}")
                    util.getUMAPOut(Y, os.path.abspath(r"../data/temp/umapOutParamDump.pkl"),
                                    n_neighbors=n_neighbors_value,
                                    min_dist=min_dist_value, n_components=n_components_value)
    print(os.path.abspath(r"../data/temp/umapOutParamDump.pkl"))
    tempUmapOut = util.getUMAPOut(Y, os.path.abspath(r"../data/temp/umapOutParamDump.pkl").replace("\\", '/'))
    if (createNewUMAP):
        umapObject = umap.UMAP(
            n_neighbors=20,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        )
        tempUmapOut = umapObject.fit_transform(Y)

    print('Umap vals: ' + str(tempUmapOut.shape))

    cell_df = pd.DataFrame(tempUmapOut, columns=['x', 'y'])
    cell_df.index = range(len(cell_df))
    matrix_legend_df.index = range(len(matrix_legend_df))
    cell_df = cell_df.merge(matrix_legend_df, left_index=True, right_index=True)
    cell_df['Task'] = cell_df['Task'].astype(str)

    # plots UMAP via Bokeh
    plotting.plotBokeh(cell_df, Y, spikePlotImagesPath, bokehShow=False)

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
