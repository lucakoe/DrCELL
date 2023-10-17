import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import umap.plot
from bokeh.events import Tap
from bokeh.io import curdoc
from bokeh.layouts import column
from colorcet import fire
from umap import plot, UMAP
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, CustomJS, Slider, Select, Checkbox, \
    MultiChoice, ColorBar, TextInput, CustomAction, TapTool
from bokeh.palettes import Spectral4, Spectral11
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


def plotBokeh(dataFrames, datas, spikePlotImagesPath, dumpFilesPaths, titles, bokehShow=True,
              startDropdownDataOption="all", debug=False, experimental=False):
    imagesPath = spikePlotImagesPath
    dataVariables=["IsChoiceSelect", "IsStimSelect", "RedNeurons", "Task"]

    for title in titles:
        dataFrames[title] = dataFrames[title].sample(frac=1, random_state=42)
        if debug: print(f"Dataframe {title}: \n{dataFrames[title]}")
        dataFrames[title]['alpha'] = 1.0

    # Create a ColumnDataSource
    datasource = ColumnDataSource(pd.DataFrame.copy(dataFrames[startDropdownDataOption]))

    # Define color mapping
    colorMapping = CategoricalColorMapper(
        factors=[str(x) for x in np.unique(dataFrames[startDropdownDataOption]['Task'])],
        palette=Spectral4)

    # Create the Bokeh figure
    plot_figure = figure(
        title='UMAP projection of Neurons',
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
            <span style='font-size: 8px'>@StimAUCs</span
            <span style='font-size: 8px; color: #224499'>Cluster:</span>
            <span style='font-size: 8px'>@Cluster</span>
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
        fill_color={'field': 'Task', 'transform': colorMapping},
        line_color={'field': 'Task', 'transform': colorMapping},
        line_alpha="alpha",
        fill_alpha="alpha",
        size=4,
        marker='circle'
    )

    # Define sliders for UMAP parameters
    if experimental:
        n_neighbors_slider = Slider(title="n_neighbors", start=1, end=50, step=1, value=20)
        min_dist_slider = Slider(title="min_dist", start=0.00, end=1.0, step=0.01, value=0.0)
    else:
        n_neighbors_slider = Slider(title="n_neighbors", start=10, end=20, step=1, value=20)
        min_dist_slider = Slider(title="min_dist", start=0.00, end=0.10, step=0.01, value=0.0)
    select_cluster_slider = Slider(title="cluster", start=-1,
                                   end=len(np.unique(dataFrames[startDropdownDataOption]['Cluster'])), step=1, value=0,
                                   disabled=True)
    min_cluster_size_slider = Slider(title="min_cluster_size", start=1, end=50, step=1, value=5, disabled=False)

    # Create a TextInput widget
    text_input = TextInput(value="0", title="Cluster:", disabled=True)


    # Define the options for the dropdown menu
    optionsSelectData = titles

    optionsFilterMultiChoiceValues={}
    for option in dataVariables:
        for value in np.unique(dataFrames[startDropdownDataOption][option]):
            optionsFilterMultiChoiceValues[f"{option} == {value}"]=(option,value)

    optionsSelectColorValues = {"Tasks": ("Task", 1)}
    optionsSelectColorValues.update(optionsFilterMultiChoiceValues.copy())


    optionsFilterMultiChoice = list(optionsFilterMultiChoiceValues.keys())
    optionsSelectColor=list(optionsSelectColorValues.keys())


    # Create the Select widget
    selectData = Select(title="Data:", value=startDropdownDataOption, options=optionsSelectData)
    selectColor = Select(title="Color:", value=optionsSelectColor[0], options=optionsSelectColor)

    # Create a MultiChoice widget with the options
    filterMultiChoice = MultiChoice(title="Filter:", value=[], options=optionsFilterMultiChoice)

    # Create a Checkbox widget
    highlightClusterCheckbox = Checkbox(label="Highlight Cluster", active=False)

    # Create a color bar for the color mapper
    color_bar = ColorBar(title="Task",color_mapper=colorMapping, location=(0, 0))
    # Add the color bar to the figure
    plot_figure.add_layout(color_bar, 'below')

    umapLayout = column(n_neighbors_slider, min_dist_slider, selectData, selectColor, filterMultiChoice,
                        min_cluster_size_slider,highlightClusterCheckbox, select_cluster_slider,  plot_figure)

    # Callback function to update UMAP when sliders change
    currentCluster=0
    datasourceDf = pd.DataFrame.copy(dataFrames[selectData.value])
    def update_umap(attr, old, new):
        n_neighbors = n_neighbors_slider.value
        min_dist = min_dist_slider.value
        nonlocal datasourceDf
        nonlocal currentCluster
        # Resets to initial state
        datasourceDf = pd.DataFrame.copy(dataFrames[selectData.value])

        umap_result = util.getUMAPOut(datas[selectData.value],
                                      os.path.abspath(dumpFilesPaths[selectData.value]).replace("\\", '/'),
                                      n_neighbors=n_neighbors, min_dist=round(min_dist, 2))
        datasourceDf['x'], datasourceDf['y'] = umap_result[:, 0], umap_result[:, 1]
        # datasource.data.update({'x': umap_result[:, 0], 'y': umap_result[:, 1]})
        if len(filterMultiChoice.value) != 0:
            for option in filterMultiChoice.value:
                datasourceDf = datasourceDf[datasourceDf[optionsFilterMultiChoiceValues[option][0]]==optionsFilterMultiChoiceValues[option][1]]

                if debug: print(type(datasourceDf[optionsFilterMultiChoiceValues[option][0]]))
                if debug: print(type(optionsFilterMultiChoiceValues[option][1]))
                if debug: print(datasourceDf[optionsFilterMultiChoiceValues[option][0]])


        if debug: print(datasourceDf)
        umap_result = datasourceDf[['x', 'y']].values

        clusters = []
        if len(umap_result) > 0:
            # Apply HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_slider.value, min_samples=1)
            clusters = clusterer.fit_predict(umap_result)

        # Add cluster labels to your dataframe
        datasourceDf['Cluster'] = clusters
        if debug: print(datasourceDf)

        select_cluster_slider.end = len(np.unique(datasourceDf['Cluster']))

        # Update the existing datasource
        datasource.data.update(ColumnDataSource(datasourceDf).data)
        updateCurrentCluster(attr=None, old=None, new=None)



        datasource.data.update(datasource.data)

    def updateCurrentCluster(attr, old, new):
        nonlocal currentCluster


        if highlightClusterCheckbox.active:
            if currentCluster != select_cluster_slider.value:
                currentCluster=select_cluster_slider.value
                text_input.value=str(currentCluster)
            elif currentCluster !=int(text_input.value):
                currentCluster=int(text_input.value)
                select_cluster_slider.value=currentCluster

            select_cluster_slider.disabled = False
            text_input.disabled=False
            print(f"Current Cluster: {currentCluster}, Current Cluster Size: {len(datasourceDf[datasourceDf['Cluster']==currentCluster])}")
            for i, cluster in enumerate(datasourceDf['Cluster']):  # Assuming cluster_data is a list of cluster labels
                if cluster == currentCluster:
                    datasource.data['alpha'][i] = 1  # Make points in the selected cluster fully visible
                else:
                    datasource.data['alpha'][i] = 0.05  # Make points in other clusters more transparent

        else:
            select_cluster_slider.disabled = True
            text_input.disabled=True

            for i, cluster in enumerate(datasourceDf['Cluster']):  # Assuming cluster_data is a list of cluster labels
                datasource.data['alpha'][i] = 1  # Make points in the selected cluster fully visible


        clustered_count = len(datasourceDf[datasourceDf['Cluster'] != -1])
        unclustered_count = len(datasourceDf[datasourceDf['Cluster'] == -1])

        # Check if the denominator (unclustered_count) is zero
        if unclustered_count == 0:
            ratio = "N/A"
        else:
            ratio = round(clustered_count / unclustered_count,3)
        # Check if the denominator (unclustered_count) is zero
        if len(datasourceDf) == 0:
            percentage = "N/A"
            clusterNumber = 0
        else:
            percentage = round((clustered_count/len(datasourceDf))*100,2)
            clusterNumber=len(np.unique(datasourceDf['Cluster'])) - 1

        print(f"Data: {selectData.value}, Filter: {filterMultiChoice.value}, Length: {len(datasourceDf)}")
        print(
            f"Clusters: {clusterNumber}, Clustered: {percentage}%, Clustered/Unclustered Ratio: {ratio}, Clustered: {clustered_count}, Unclustered: {unclustered_count}"
        )

        #Change Color
        nonlocal colorMapping

        colorMapping = CategoricalColorMapper(factors=[str(x) for x in np.unique(datasourceDf[optionsSelectColorValues[selectColor.value][0]])],palette=Spectral4)
        datasource.data['fill_color']={'field': optionsSelectColorValues[selectColor.value][0], 'transform': colorMapping}
        datasource.data['line_color']={'field': optionsSelectColorValues[selectColor.value][0], 'transform': colorMapping}
        color_bar.color_mapper=colorMapping








        print("\n")

        datasource.data.update(datasource.data)

    # Define a Python callback to handle point clicks
    def on_point_click(event):
        return
        # if not datasource.selected.indices:
        #     print("No point")
        #     return  # No point was clicked
        #
        # index = datasource.selected.indices[0]
        # clicked_label = datasource.data['Neuron'][index]
        # print(f"Point {clicked_label} was clicked!")

    # Attach the callback function to slider changes
    n_neighbors_slider.on_change('value', update_umap)
    min_dist_slider.on_change('value', update_umap)
    select_cluster_slider.on_change('value', updateCurrentCluster)
    min_cluster_size_slider.on_change('value', update_umap)
    # Attach the callback function to the dropdown menu
    selectData.on_change('value', update_umap)
    selectColor.on_change('value', update_umap)

    text_input.on_change('value', updateCurrentCluster)
    # Attach the callback function to the MultiChoice's "value" property
    filterMultiChoice.on_change('value', update_umap)
    # Attach the callback to the checkbox
    highlightClusterCheckbox.on_change('active', updateCurrentCluster)

    plot_figure.on_event(Tap, on_point_click)


    # Create a layout for the sliders and plot
    layout = column(umapLayout)

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
