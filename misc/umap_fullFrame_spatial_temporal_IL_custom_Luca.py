# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:42:38 2020

@author: Simon
"""
import io
import pickle

import numpy as np
import pandas
# import h5py
import umap
# import hdbscan
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, CustomJS
from bokeh.palettes import Spectral4

import util

# skips UMAP creation if false and takes dump
createUMAP = False
# skips Spike Plot generation if false and takes dump
generatePlots = False
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

    # Extract the 'UMAPmatrix' array from the loaded data
    umap_matrix = g['UMAPmatrix']

    # Create a Pandas DataFrame from 'UMAPmatrix'
    umap_df = pd.DataFrame(umap_matrix)

    # Extract the 'matrixLegendArray' array from the loaded data
    matrix_legend_array = g['matrixLegendArray']

    # Create a Pandas DataFrame from 'matrixLegendArray'
    matrix_legend_df = pd.DataFrame(matrix_legend_array,
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
    array_of_plots = []
    if (generatePlots):
        for i in range(umap_df.shape[0]):
            # takes selected row (fluorescence data of one cell), makes it to an array and plots it
            plt = util.plot_spikes(umap_df.iloc[i].values)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80)
            buf.seek(0)
            array_of_plots.append(buf.read())
            plt.close()
            print(i)

        with open(r"../data/temp/array_of_plots.pkl", 'wb') as file:
            pickle.dump(array_of_plots, file)

    with open(r"../data/temp/array_of_plots.pkl", 'rb') as file:
        array_of_plots = pickle.load(file)

    tempUmapOut = pandas.DataFrame
    if (createUMAP):
        tempUmapOut = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(Y)
        print('Umap vals: ' + str(tempUmapOut.shape))

        # for debugging, dumps data to save time
        with open(r"../data/temp/tempUmapOut.pkl", 'wb') as file:
            pickle.dump(tempUmapOut, file)

    with open(r"../data/temp/tempUmapOut.pkl", 'rb') as file:
        tempUmapOut = pickle.load(file)

    cell_df = pd.DataFrame(tempUmapOut, columns=['x', 'y'])
    cell_df.index = range(len(cell_df))
    matrix_legend_df.index = range(len(matrix_legend_df))
    cell_df = cell_df.merge(matrix_legend_df, left_index=True, right_index=True)
    cell_df['Task'] = cell_df['Task'].astype(str)
    # cell_df["Plot"]= array_of_plots
    # Create a ColumnDataSource
    datasource = ColumnDataSource(cell_df)
    # Define color mapping
    color_mapping = CategoricalColorMapper(factors=[str(x) for x in np.unique(cell_df['Task'])],
                                           palette=Spectral4)

    # Create the Bokeh figure
    plot_figure = figure(
        title='UMAP projection of Cells',
        width=600,  # Use 'width' instead of 'plot_width'
        height=600,  # Use 'height' instead of 'plot_height'
        tools='pan, wheel_zoom, reset'
    )

    # Add a HoverTool to display the Matplotlib plot when hovering over a data point
    plot_figure.add_tools(HoverTool(tooltips="""
        <div>
            <span style='font-size: 8px; color: #224499'>Neuron:</span>
            <span style='font-size: 8px'>@Neuron</span>
            <span style='font-size: 8px; color: #224499'>ChoiceAUCs:</span>
            <span style='font-size: 8px'>@ChoiceAUCs</span>
            <span style='font-size: 8px; color: #224499'>StimAUCs:</span>
            <span style='font-size: 8px'>@StimAUCs</span>
        </div>
        
        <div>
            <img
                src="file:///C:/Users/koenig/Documents/GitHub/twoP/Playground/Luca/PlaygoundProject/data/temp/plot_images/image_@{index}.png" height="100" alt="Image"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            />
    </div>
    """))

    # Create a scatter plot
    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        fill_color={'field': 'Task', 'transform': color_mapping},
        line_color={'field': 'Task', 'transform': color_mapping},
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )

    # Define a JavaScript callback to open the Matplotlib plot when a data point is clicked
    callback_code = """
    var data_index = cb_data.index['1d'].indices[0];
    var data_url = 'data:image/png;base64,' + btoa(array_of_plots[data_index]);
    window.open(data_url, '_blank');
    """

    # Add the JavaScript callback to the plot
    plot_figure.js_on_event('tap', CustomJS(code=callback_code))

    # Show the plot
    show(plot_figure)

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
