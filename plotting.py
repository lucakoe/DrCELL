import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import umap.plot
from bokeh.io import curdoc
from bokeh.layouts import column
from umap import plot, UMAP
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, CustomJS, Slider, Select
from bokeh.palettes import Spectral4
import io
import os

import util


def generateDiagnosticPlots(umapObject, data):
    mapper = umapObject.fit(data)
    umap.plot.diagnostic(mapper, diagnostic_type='pca')
    plt.show()
    umap.plot.diagnostic(mapper, diagnostic_type='vq')
    plt.show()
    umap.plot.diagnostic(mapper, diagnostic_type='local_dim')
    plt.show()
    umap.plot.diagnostic(mapper, diagnostic_type='neighborhood')
    plt.show()


def plotBokeh(dataFrames, datas, spikePlotImagesPath, dumpFilesPaths, titles, bokehShow=True, startDropdownDataOption="all"):

    imagesPath = spikePlotImagesPath

    for title in titles:
        dataFrames[title] = dataFrames[title].sample(frac=1, random_state=42)

    # Create a ColumnDataSource
    datasource = ColumnDataSource(pd.DataFrame.copy(dataFrames[startDropdownDataOption]))

    # Define color mapping
    color_mapping = CategoricalColorMapper(factors=[str(x) for x in np.unique(dataFrames[startDropdownDataOption]['Task'])],
                                           palette=Spectral4)

    # Create the Bokeh figure
    plot_figure = figure(
        title='UMAP projection of Cells',
        width=600,
        height=600,
        tools='pan, wheel_zoom, box_zoom,save, reset, help'
    )

    # Add a HoverTool to display the Matplotlib plot when hovering over a data point
    plot_figure.add_tools(HoverTool(tooltips=f"""
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
                src="file://{imagesPath}""" + """/image_@{index}.png" height="100" alt="Image"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            />
    </div>
    """))

    # Create a scatter plot
    plot_figure.scatter(
        'x',
        'y',
        source=datasource,
        fill_color={'field': 'Task', 'transform': color_mapping},
        line_color={'field': 'Task', 'transform': color_mapping},
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4,
        marker='circle'
    )

    # Define sliders for UMAP parameters
    n_neighbors_slider = Slider(title="n_neighbors", start=10, end=20, step=1, value=20)
    min_dist_slider = Slider(title="min_dist", start=0.00, end=0.10, step=0.01, value=0.0)

    # Define the options for the dropdown menu
    optionsSelectData = titles
    optionsSelectFilter = ["None", "IsChoiceSelect", "IsStimSelect"]

    # Create the Select widget
    selectData = Select(title="Data:", value=startDropdownDataOption, options=optionsSelectData)
    selectFilter = Select(title="Filter:", value="None", options=optionsSelectFilter)

    # Callback function to update UMAP when sliders change

    def update_umap(attr, old, new):
        n_neighbors = n_neighbors_slider.value
        min_dist = min_dist_slider.value

        # Resets to initial state
        datasourceDf=pd.DataFrame.copy(dataFrames[selectData.value])

        umap_result = util.getUMAPOut(datas[selectData.value], os.path.abspath(dumpFilesPaths[selectData.value]).replace("\\", '/'),
                                      n_neighbors=n_neighbors, min_dist=round(min_dist, 2))
        datasourceDf['x'],datasourceDf['y']=umap_result[:, 0],umap_result[:, 1]
        # datasource.data.update({'x': umap_result[:, 0], 'y': umap_result[:, 1]})
        if not (selectFilter.value == 'None'):
            datasourceDf=datasourceDf[datasourceDf[selectFilter.value]]
            print(f"Data:{selectData.value} Filter: {selectFilter.value}, Length: {len(datasourceDf[selectFilter.value])} ")
        else:
            print(f"Data:{selectData.value} Filter: {selectFilter.value}, Length: {len(datasourceDf[optionsSelectFilter[1]])}")
        # Update the existing datasource
        datasource.data.update(ColumnDataSource(datasourceDf).data)

    # Attach the callback function to slider changes
    n_neighbors_slider.on_change('value', update_umap)
    min_dist_slider.on_change('value', update_umap)
    # Attach the callback function to the dropdown menu
    selectData.on_change('value', update_umap)
    selectFilter.on_change('value', update_umap)

    # Create a layout for the sliders and plot
    layout = column(n_neighbors_slider, min_dist_slider, selectData, selectFilter, plot_figure)

    if bokehShow:
        # Show the plot
        show(layout)
    else:
        # for Bokeh Server
        curdoc().add_root(layout)


def plotAndReturnSpikes(fluorescence_array, fps=30):
    # interactive view
    number_consecutive_recordings = 6
    # Calculate time values based on the frame rate (30 fps)
    fluorescence_array

    n = len(fluorescence_array)
    time_values = np.arange(n) / fps

    # Plot intensity against time
    plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
    plt.plot(time_values, fluorescence_array, linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence Intensity')
    plt.title('Fluorescence Intensity vs. Time')
    plt.grid(True)

    # Show the plot
    # plt.show()

    return plt


def plotAndSaveSpikes(neuronNumber, dataframe, outputFolder):
    # takes selected row (fluorescence data of one cell), makes it to an array and plots it
    plt = plotAndReturnSpikes(dataframe.iloc[neuronNumber].values)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80)
    buf.seek(0)
    # Ensure the output directory exists; create it if it doesn't
    os.makedirs(outputFolder, exist_ok=True)

    # Save each image as a separate PNG file
    filePath = os.path.join(outputFolder, f"image_{neuronNumber}.png")
    with open(filePath, "wb") as file:
        file.write(buf.read())

    print(f"Saved image {neuronNumber} to {filePath}")
    plt.close()
