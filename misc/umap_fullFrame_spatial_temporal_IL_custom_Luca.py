# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:42:38 2020

@author: Simon
"""
import os
import sys

import hdbscan

projectPath = r"C:\Users\koenig\Documents\GitHub\twoP\Playground\Luca\PlaygoundProject"
# change working directory, because Bokeh Server doesnt recognize it otherwise
os.chdir(os.path.join(projectPath, "misc"))
# add to project Path so Bokeh Server can import other python files correctly
sys.path.append(projectPath)

import numpy as np
import pandas
# import h5py
import umap
# import hdbscan

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotting
import util

# skips Spike Plot generation if false and takes dump
generateAndSaveSpikePlots = False
generateDiagnosticPlots = False
generateUMAPParameters = False
debugging = False
spikePlotImagesPath = r'/C:/Users/koenig/Documents/GitHub/twoP/Playground/Luca/PlaygoundProject/data/temp/plot_images'
dumpFilesPath = r"../data/temp/"
# spikePlotImagesPath=os.path.join(spikePlotImagesPath, title)
umapOutParamDumpFilenameExtention = "_umapOutParamDump.pkl"
bokehShow = False
allFiles = [r'../data/Umap_2530_2532_Array.mat']
outputPath = r'../data'
# ['C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate2530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixTaskWithoutDelay2530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixTaskWithDelay2530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate22530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate2532.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixAudioTaskWithoutDelay2532.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixAudioTaskWithDelay2532.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate22532.mat']
# 
# allFiles = ['Q:\\BpodImager\\umapClust\\20210309T2356_fullA.mat'] #first file

if __name__ == '__main__':
    bokehShow = True

for cFile in allFiles:
    titles = ["all", "excludeChoiceUnselectBefore", "excludeStimUnselectBefore"]
    umapDf, matrixLegendDf, cleanedData = util.loadAndPreprocessData(cFile)

    cleanedDatas = {}
    matrixLegendDfs = {}
    dumpFilesPaths = {}
    cellDfs = {}

    for title in titles:
        cleanedDatas[title] = cleanedData
        matrixLegendDfs[title] = matrixLegendDf

        if title == "all":
            print(f"{title} Data Length: {len(matrixLegendDf)}")

        elif title == "excludeChoiceUnselectBefore":
            # Filters cells with Property
            cleanedDatas[title] = cleanedData[matrixLegendDf["IsChoiceSelect"]]
            matrixLegendDfs[title] = matrixLegendDf[matrixLegendDf["IsChoiceSelect"]]
            print(f"{title} Data Length: {matrixLegendDf['IsChoiceSelect'].apply(lambda x: x).sum()}")

        elif title == "excludeStimUnselectBefore":
            # Filters cells with Property
            cleanedDatas[title] = cleanedData[matrixLegendDf["IsStimSelect"]]
            matrixLegendDfs[title] = matrixLegendDf[matrixLegendDf["IsStimSelect"]]
            print(f"{title} Data Length: {matrixLegendDf['IsStimSelect'].apply(lambda x: x).sum()}")


        if debugging: print(f"Cleaned Data {title}: \n{cleanedDatas[title]}")

        dumpFilesPaths[title] = os.path.abspath(os.path.join(dumpFilesPath, title + umapOutParamDumpFilenameExtention))

        # Convert the Matplotlib plots to base64-encoded PNGs

        if (generateAndSaveSpikePlots):
            for i in range(umapDf.shape[0]):
                plotting.plotAndSaveSpikes(i, umapDf, spikePlotImagesPath)

        tempUmapOut = pandas.DataFrame
        umapObject = umap.UMAP

        if generateUMAPParameters:
            n_neighbors_values = range(10, 21, 1)
            min_dist_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
            n_components_values = [2]
            for n_neighbors_value in n_neighbors_values:
                for min_dist_value in min_dist_values:
                    for n_components_value in n_components_values:
                        util.getUMAPOut(cleanedDatas[title],
                                        os.path.abspath(dumpFilesPaths[title]),
                                        n_neighbors=n_neighbors_value,
                                        min_dist=min_dist_value, n_components=n_components_value)

        # gets the UMAP Output file. This function is used to buffer already created UMAPs and improve performance
        print(os.path.abspath(dumpFilesPaths[title]))
        tempUmapOut = util.getUMAPOut(cleanedDatas[title],
                                      os.path.abspath(dumpFilesPaths[title]).replace(
                                          "\\", '/'),
                                      n_neighbors=20, min_dist=0.0, n_components=2)

        print('Umap vals: ' + str(tempUmapOut.shape))

        # Apply HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1)
        clusters = clusterer.fit_predict(tempUmapOut)

        cellDfs[title] = pd.DataFrame(tempUmapOut, columns=['x', 'y'])
        # creates an index for merging
        cellDfs[title].index = range(len(cellDfs[title]))
        matrixLegendDfs[title].index = range(len(matrixLegendDfs[title]))
        cellDfs[title] = cellDfs[title].merge(matrixLegendDfs[title], left_index=True, right_index=True)
        cellDfs[title]['Task'] = cellDfs[title]['Task'].astype(str)
        # Add cluster labels to your dataframe
        cellDfs[title]['Cluster'] = clusters

    # plots UMAP via Bokeh
    plotting.plotBokeh(cellDfs, cleanedDatas, spikePlotImagesPath, dumpFilesPaths, titles, bokehShow=bokehShow,
                       startDropdownDataOption=titles[0])

    # # show results
    # cIdx = np.random.permutation(matrixLegendDf["Task"].size)
    # fig1, ax1 = plt.subplots()
    # # cColor = sns.color_palette("Dark2", np.max([groups])+1);
    # cColor = sns.color_palette("Spectral", np.max([matrixLegendDf["Task"].astype(float).astype(int)]) + 1);
    # ax1.scatter(tempUmapOut[cIdx, 0], tempUmapOut[cIdx, 1], c=[cColor[x] for x in matrixLegendDf["Task"][cIdx]], s=2)
    # # plt.axis('square')
    # ax1.set_aspect('equal', adjustable='datalim')
    # plt.show()

    # sio.savemat(cFile.replace('.mat','_umap_allAreas.mat'),
    #                 {'umapOut':umapOut,
    #                  'tempUmapOut':tempUmapOut,
    #                  })

    # Save UMAP coordinates and labels to see in Matlab
    # umap_data = np.column_stack((tempUmapOut, matrixLegendDf["Task"]))
    # file_name = os.path.basename(cFile)
    # sio.savemat(r'\\Naskampa\lts\2P_PuffyPenguin\2P_PuffyPenguin\Umap_Analysis\output_' + file_name, {'umap_data': umap_data})

    if generateDiagnosticPlots:
        (plotting.generateDiagnosticPlots(umapObject, cleanedData))
