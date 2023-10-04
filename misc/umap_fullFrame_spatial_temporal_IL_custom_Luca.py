# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:42:38 2020

@author: Simon
"""

import numpy as np
#import h5py
import umap
#import hdbscan
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10


import util

allFiles = [r'../data/Umap_2530_2532_Array.mat']
outputPath = r'../data'
#['C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate2530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixTaskWithoutDelay2530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixTaskWithDelay2530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate22530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate2532.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixAudioTaskWithoutDelay2532.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixAudioTaskWithDelay2532.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate22532.mat']
# 
# allFiles = ['Q:\\BpodImager\\umapClust\\20210309T2356_fullA.mat'] #first file


for cFile in allFiles:
    # load data
    print(cFile)
    g = sio.loadmat(cFile)
    #g = h5py.File(cFile,'r')
    #X = np.array(g['X'])
    X = np.array(g['UMAPmatrix'])
    Y = np.array(g['UMAPmatrix'])
    # Y = np.array(g['Y'])
    #Y = np.transpose(Y)
    matrixLegend = np.array(g['matrixLegendArray'])
    # groups = np.array(g['redIdxAll'])
    # groups = np.reshape(groups,X.shape[0])
    # groups = groups.astype(int)
    
    #groups = np.array(g['groupIdx'])


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
    
    #g.close

    # Extract the 'UMAPmatrix' array from the loaded data
    umap_matrix = g['UMAPmatrix']

    # Create a Pandas DataFrame from 'UMAPmatrix'
    umap_df = pd.DataFrame(umap_matrix)

    # Extract the 'matrixLegendArray' array from the loaded data
    matrix_legend_array = g['matrixLegendArray']

    # Create a Pandas DataFrame from 'matrixLegendArray'
    matrix_legend_df = pd.DataFrame(matrix_legend_array)
    matrix_legend_df.columns = ["Animal", "Recording", "Neuron", "NbFrames", "Task", "RedNeurons", "ChoiceAUCs", "IsChoiceSelect", "StimAUCs", "IsStimSelect"]
    matrix_legend_df['Task']=matrix_legend_df['Task'].astype(int)

    useIdx = np.invert(np.isnan(np.sum(Y, axis=0)))
    Y = Y[:,useIdx]
    
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
    tempUmapOut = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
        ).fit_transform(Y)
    print('Umap vals: ' + str(tempUmapOut.shape))
    
    # show results
    cIdx = np.random.permutation(matrix_legend_df["Task"].size)
    fig1, ax1 = plt.subplots()
    # cColor = sns.color_palette("Dark2", np.max([groups])+1);
    cColor = sns.color_palette("Spectral", np.max([matrix_legend_df["Task"]])+1);
    ax1.scatter(tempUmapOut[cIdx, 0], tempUmapOut[cIdx, 1], c=[cColor[x] for x in matrix_legend_df["Task"][cIdx]], s=2)
    #plt.axis('square')
    ax1.set_aspect('equal', adjustable = 'datalim')
    plt.show() 
    
    # sio.savemat(cFile.replace('.mat','_umap_allAreas.mat'),
    #                 {'umapOut':umapOut,
    #                  'tempUmapOut':tempUmapOut,
    #                  })
    
    # Save UMAP coordinates and labels to see in Matlab
    umap_data = np.column_stack((tempUmapOut, matrix_legend_df["Task"]))
    file_name = os.path.basename(cFile)
    sio.savemat(r'\\Naskampa\lts\2P_PuffyPenguin\2P_PuffyPenguin\Umap_Analysis\output_' + file_name, {'umap_data': umap_data})