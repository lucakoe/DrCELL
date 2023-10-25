import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import umap.plot
from bokeh.events import Tap
from bokeh.io import curdoc
from bokeh.layouts import column, row
from colorcet import fire
from umap import plot, UMAP
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, CustomJS, Slider, Select, Checkbox, \
    MultiChoice, ColorBar, TextInput, CustomAction, TapTool, CheckboxGroup, RadioGroup, Div, Button
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
    dataVariables = ["IsChoiceSelect", "IsStimSelect", "RedNeurons", "Task"]
    displayHoverVariables = ["Neuron", "ChoiceAUCs", "StimAUCs", "Cluster"]

    for title in titles:
        dataFrames[title] = dataFrames[title].sample(frac=1, random_state=42)
        if debug: print(f"Dataframe {title}: \n{dataFrames[title]}")
        dataFrames[title]['alpha'] = 1.0

    datasourceDf = pd.DataFrame.copy(dataFrames[startDropdownDataOption])
    # Create a ColumnDataSource
    datasource = ColumnDataSource(datasourceDf)

    # Define color mapping
    colorMapping = CategoricalColorMapper(
        factors=[str(x) for x in np.unique(datasourceDf['Task'])],
        palette=Spectral4)

    # Create the Bokeh figure
    plot_figure = figure(
        title='UMAP projection of Neurons',
        width=600,
        height=600,
        tools='pan, wheel_zoom, box_zoom,save, reset, help',
        toolbar_location="right"
    )

    # Create gid

    # Grid adjustments
    gridSizeX = 1.0
    gridSizeY = 1.0
    gridStartPos = (0.0, 0.0)
    minPoint = (datasourceDf["x"].min(), datasourceDf["y"].min())
    maxPoint = (datasourceDf["x"].max(), datasourceDf["y"].max())

    # Generate data points in the middle of each grid cell
    gridDatasourceDf = util.generateGrid(minPoint, maxPoint, centerPoint=gridStartPos,
                                         grid_size_x=gridSizeX, grid_size_y=gridSizeY)

    gridDatasourceDf['alpha'] = 0.1
    gridDatasourceDf['gridSizeX'] = gridSizeX
    gridDatasourceDf['gridSizeY'] = gridSizeY
    # gridDatasourceDf = util.assign_points_to_grid(datasourceDf, gridDatasourceDf)
    gridDatasource = ColumnDataSource(gridDatasourceDf)

    # Create a Bokeh plot
    gridPlot = plot_figure.rect('centerX', 'centerY', 'gridSizeX', 'gridSizeY',
                                source=gridDatasource,
                                fill_color="lightblue",
                                line_color="black",
                                line_alpha='alpha',
                                fill_alpha='alpha')


    # Create a scatter plot
    scatterPlot = plot_figure.scatter(
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

    # Custom Tools

    hoverVariableString = ""
    for variable in displayHoverVariables:
        hoverVariableString += f"""<span style='font-size: 8px; color: #224499'>{variable}:</span>\n
                <span style='font-size: 8px'>@{variable}</span>\n"""

    scatterPlotHoverTool = HoverTool(tooltips=f"""
        <div>
            {hoverVariableString}
        </div>

        <div>
            <img
                src="file://{imagesPath}""" + """/image_@{index}.png" height="100" alt="Image"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            />
    </div>
    """, renderers=[scatterPlot])

    gridPlotHoverTool = HoverTool(tooltips="""
    <div>
        Grid ID: @{gridID}
        Grid Point Indices: @{pointIndices}
        Grid Point Neurons: @{pointNeurons}
    </div>
    <div>
        <img src="data:image/png;base64, @{image}" style="float: left; margin: 0px 15px 15px 0px; width: 100px; height: auto;">

    </div>
           
    </div>
    """, renderers=[gridPlot])

    # Add a HoverTool to display the Matplotlib plot when hovering over a data point
    plot_figure.add_tools(scatterPlotHoverTool)
    plot_figure.add_tools(gridPlotHoverTool)

    # Create an empty line Div for spacing
    blankDiv = Div(text="<br>", width=400, height=5)

    # General

    generalTitleDiv = Div(text="<h3>General: </h3>", width=400, height=20)

    optionsSelectData = titles
    selectData = Select(title="Data:", value=startDropdownDataOption, options=optionsSelectData)

    optionsFilterMultiChoiceValues = {}
    for option in dataVariables:
        for value in np.unique(dataFrames[startDropdownDataOption][option]):
            optionsFilterMultiChoiceValues[f"{option} == {value}"] = (option, value)
    optionsSelectColorValues = {"Tasks": ("Task", 1)}
    optionsSelectColorValues.update(optionsFilterMultiChoiceValues.copy())
    optionsSelectColor = list(optionsSelectColorValues.keys())
    selectColor = Select(title="Color:", value=optionsSelectColor[0], options=optionsSelectColor)

    optionsFilterMultiChoice = list(optionsFilterMultiChoiceValues.keys())
    orFilterMultiChoice = MultiChoice(title="'OR' Filter:", value=[], options=optionsFilterMultiChoice, width=200)
    andFilterMultiChoice = MultiChoice(title="'AND' Filter:", value=[], options=optionsFilterMultiChoice, width=200)

    generalLayout = column(generalTitleDiv, blankDiv, selectData, selectColor, orFilterMultiChoice,
                           andFilterMultiChoice)

    # Hover Tool and Grid selection
    enableGridCheckbox = Checkbox(label="Grid Enabled", active=False)
    gridPlot.visible = enableGridCheckbox.active
    gridSizeXTextInput = TextInput(value="1.0", title="Grid Size X:", disabled=False)
    gridSizeYTextInput = TextInput(value="1.0", title="Grid Size Y:", disabled=False)
    gridSizeButton = Button(label="Update")
    hoverToolLayout = column(enableGridCheckbox, gridSizeXTextInput, gridSizeYTextInput, gridSizeButton)

    # UMAP
    umapTitleDiv = Div(text="<h3>UMAP Parameters: </h3>", width=400, height=20)

    if experimental:
        n_neighbors_slider = Slider(title="n_neighbors", start=2, end=50, step=1, value=20)
        min_dist_slider = Slider(title="min_dist", start=0.00, end=1.0, step=0.01, value=0.0)
    else:
        n_neighbors_slider = Slider(title="n_neighbors", start=10, end=20, step=1, value=20)
        min_dist_slider = Slider(title="min_dist", start=0.00, end=0.10, step=0.01, value=0.0)

    umapLayout = column(umapTitleDiv, blankDiv, n_neighbors_slider, min_dist_slider)

    # Cluster Parameters

    clusterParametersTitleDiv = Div(text="<h3>Cluster Parameters: </h3>", width=400, height=20)

    min_cluster_size_slider = Slider(title="min_cluster_size", start=1, end=50, step=1, value=5, disabled=False)
    min_samples_slider = Slider(title="min_sample", start=1, end=10, step=1, value=1, disabled=False)
    optionsClusterSelectionMethod = ['eom', 'leaf']
    clusterSelectionMethodToggle = RadioGroup(labels=optionsClusterSelectionMethod,
                                              active=0)  # Set "eom" as the default
    cluster_selection_epsilon_slider = Slider(title="cluster_selection_epsilon", start=0.00, end=1.0, step=0.01,
                                              value=0.0)
    optionsMetric = ["euclidean", "manhattan", "correlation", "jaccard", "hamming", "chebyshev", "canberra",
                     "braycurtis"]
    selectMetric = Select(title="metric:", value=optionsMetric[0], options=optionsMetric)

    allowSingleLinkageToggle = Checkbox(label="Allow Single-Linkage", active=False)
    approximateMinimumSpanningTreeToggle = Checkbox(label="Approximate Minimum Spanning Tree", active=True)

    clusterParametersLayout = column(clusterParametersTitleDiv, blankDiv, min_cluster_size_slider, min_samples_slider,
                                     cluster_selection_epsilon_slider,
                                     allowSingleLinkageToggle, approximateMinimumSpanningTreeToggle, selectMetric,
                                     clusterSelectionMethodToggle)

    # Cluster Selection

    clusterSelectionTitleDiv = Div(text="<h3>Cluster Selection: </h3>", width=400, height=20)

    highlightClusterCheckbox = Checkbox(label="Highlight Cluster", active=False)
    selectedClusterTextInput = TextInput(value="0", title="Cluster:", disabled=True)
    select_cluster_slider = Slider(title="cluster", start=-1,
                                   end=len(np.unique(dataFrames[startDropdownDataOption]['Cluster'])), step=1, value=0,
                                   disabled=True)
    clusterSelectionLayout = column(clusterSelectionTitleDiv, blankDiv, highlightClusterCheckbox,
                                    selectedClusterTextInput,
                                    select_cluster_slider)

    # Create a color bar for the color mapper
    color_bar = ColorBar(title="Task", color_mapper=colorMapping, location=(0, 0))
    # Add the color bar to the figure
    plot_figure.add_layout(color_bar, 'below')

    mainLayoutTitleDiv = Div(text="<h2>Cluster Exploration and Labeling Library: </h2>", width=800, height=20)

    mainLayoutRow = row(generalLayout, column(umapLayout, clusterSelectionLayout), clusterParametersLayout)
    mainLayout = column(mainLayoutTitleDiv, blankDiv,
                        mainLayoutRow,
                        row(plot_figure, hoverToolLayout))

    currentCluster = 0

    # Callback function to update graph when sliders change
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
        if len(orFilterMultiChoice.value) != 0:
            datasourceDf = pd.DataFrame(columns=datasourceDf.columns)
            for option in orFilterMultiChoice.value:
                initialDf = pd.DataFrame.copy(dataFrames[selectData.value])
                # makes a dataframe with just the filtered entrys and merges it with the other slected values
                filterDf = initialDf[
                    initialDf[optionsFilterMultiChoiceValues[option][0]] == optionsFilterMultiChoiceValues[option][
                        1]]
                datasourceDf = pd.merge(datasourceDf, filterDf, how='outer')

            datasourceDf = datasourceDf.drop_duplicates(keep="first")
            # shuffles Data order for plotting
            datasourceDf = datasourceDf.sample(frac=1, random_state=42)

        if len(andFilterMultiChoice.value) != 0:
            for option in andFilterMultiChoice.value:
                datasourceDf = datasourceDf[
                    datasourceDf[optionsFilterMultiChoiceValues[option][0]] == optionsFilterMultiChoiceValues[option][
                        1]]

                if debug: print(type(datasourceDf[optionsFilterMultiChoiceValues[option][0]]))
                if debug: print(type(optionsFilterMultiChoiceValues[option][1]))
                if debug: print(datasourceDf[optionsFilterMultiChoiceValues[option][0]])

        if debug: print(datasourceDf)
        umap_result = datasourceDf[['x', 'y']].values

        clusters = []
        if len(umap_result) > 0:
            # Apply HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_slider.value,
                                        min_samples=min_samples_slider.value,
                                        allow_single_cluster=allowSingleLinkageToggle.active,
                                        approx_min_span_tree=approximateMinimumSpanningTreeToggle.active,
                                        cluster_selection_method=optionsClusterSelectionMethod[
                                            clusterSelectionMethodToggle.active],
                                        metric=selectMetric.value,
                                        cluster_selection_epsilon=cluster_selection_epsilon_slider.value)
            clusters = clusterer.fit_predict(umap_result)

        # Add cluster labels to your dataframe
        datasourceDf['Cluster'] = clusters
        if debug: print(datasourceDf)

        select_cluster_slider.end = len(np.unique(datasourceDf['Cluster']))

        # Update the existing datasource
        datasource.data.update(ColumnDataSource(datasourceDf).data)
        updateCurrentCluster(attr=None, old=None, new=None)
        updateGrid(attr=None, old=None, new=None)
        datasource.data.update(datasource.data)

    def updateCurrentCluster(attr, old, new):
        nonlocal currentCluster

        if highlightClusterCheckbox.active:
            if currentCluster != select_cluster_slider.value:
                currentCluster = select_cluster_slider.value
                selectedClusterTextInput.value = str(currentCluster)
            elif currentCluster != int(selectedClusterTextInput.value):
                currentCluster = int(selectedClusterTextInput.value)
                select_cluster_slider.value = currentCluster

            select_cluster_slider.disabled = False
            selectedClusterTextInput.disabled = False
            print(
                f"Current Cluster: {currentCluster}, Current Cluster Size: {len(datasourceDf[datasourceDf['Cluster'] == currentCluster])}")
            for i, cluster in enumerate(datasourceDf['Cluster']):  # Assuming cluster_data is a list of cluster labels
                if cluster == currentCluster:
                    datasource.data['alpha'][i] = 1  # Make points in the selected cluster fully visible
                else:
                    datasource.data['alpha'][i] = 0.05  # Make points in other clusters more transparent

        else:
            select_cluster_slider.disabled = True
            selectedClusterTextInput.disabled = True

            for i, cluster in enumerate(datasourceDf['Cluster']):  # Assuming cluster_data is a list of cluster labels
                datasource.data['alpha'][i] = 1  # Make points in the selected cluster fully visible

        clustered_count = len(datasourceDf[datasourceDf['Cluster'] != -1])
        unclustered_count = len(datasourceDf[datasourceDf['Cluster'] == -1])

        # Check if the denominator (unclustered_count) is zero
        if unclustered_count == 0:
            ratio = "N/A"
        else:
            ratio = round(clustered_count / unclustered_count, 3)
        # Check if the denominator (unclustered_count) is zero
        if len(datasourceDf) == 0:
            percentage = "N/A"
            clusterNumber = 0
        else:
            percentage = round((clustered_count / len(datasourceDf)) * 100, 2)
            clusterNumber = len(np.unique(datasourceDf['Cluster'])) - 1

        print(
            f"Data: {selectData.value}, 'AND' Filter: {andFilterMultiChoice.value}, 'OR' Filter: {orFilterMultiChoice.value}, Length: {len(datasourceDf)}")
        print(
            f"Clusters: {clusterNumber}, Clustered: {percentage}%, Clustered/Unclustered Ratio: {ratio}, Clustered: {clustered_count}, Unclustered: {unclustered_count}"
        )

        # Change Color
        nonlocal colorMapping

        # colorMapping = CategoricalColorMapper(factors=[str(x) for x in np.unique(datasourceDf[optionsSelectColorValues[selectColor.value][0]])],palette=Spectral4)
        # datasource.data['fill_color']={'field': optionsSelectColorValues[selectColor.value][0], 'transform': colorMapping}
        # datasource.data['line_color']={'field': optionsSelectColorValues[selectColor.value][0], 'transform': colorMapping}
        # color_bar.color_mapper=colorMapping

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

    def updateGridButton():
        updateGrid(attr=None, old=None, new=None)
    def updateGrid(attr, old, new):
        global gridSizeY, gridSizeX
        if enableGridCheckbox.active:
            gridSizeX = float(gridSizeXTextInput.value)
            gridSizeY = float(gridSizeYTextInput.value)
            gridStartPos = (0.0, 0.0)
            minPoint = (datasourceDf["x"].min(), datasourceDf["y"].min())
            maxPoint = (datasourceDf["x"].max(), datasourceDf["y"].max())

            # Generate data points in the middle of each grid cell
            gridDatasourceDf = util.generateGrid(minPoint, maxPoint, centerPoint=gridStartPos,
                                                 grid_size_x=gridSizeX, grid_size_y=gridSizeY)
            gridDatasourceDf['gridSizeX'] = gridSizeX
            gridDatasourceDf['gridSizeY'] = gridSizeY
            gridDatasourceDf['alpha'] = 0.1
            gridDatasourceDf = util.assign_points_to_grid(datasourceDf, gridDatasourceDf, [('index','pointIndices'), ("Neuron","pointNeurons")])

            gridDatasource.data.update(ColumnDataSource(gridDatasourceDf).data)

        gridPlot.visible=enableGridCheckbox.active

    def hover_callback(attr, old_index, new_index):
        if new_index:
            selected_data = gridDatasource.data
            selected_x = selected_data['x'][new_index[0]]
            selected_y = selected_data['y'][new_index[0]]
            selected_index = selected_data['index'][new_index[0]]
            print(f"Hovered over data point at x={selected_x}, y={selected_y}, index={selected_index}")

    # Attach the callback function to slider changes
    n_neighbors_slider.on_change('value_throttled', update_umap)
    min_dist_slider.on_change('value_throttled', update_umap)
    min_samples_slider.on_change('value_throttled', update_umap)
    cluster_selection_epsilon_slider.on_change('value_throttled', update_umap)
    select_cluster_slider.on_change('value_throttled', updateCurrentCluster)
    min_cluster_size_slider.on_change('value_throttled', update_umap)
    # Attach the callback function to the dropdown menu
    selectData.on_change('value', update_umap)
    selectMetric.on_change('value', update_umap)
    selectColor.on_change('value', update_umap)
    selectedClusterTextInput.on_change('value', updateCurrentCluster)
    gridSizeButton.on_click(updateGridButton)
    enableGridCheckbox.on_change('active', updateGrid)

    # Attach the callback function to the CheckboxGroup's active property
    clusterSelectionMethodToggle.on_change("active", update_umap)
    allowSingleLinkageToggle.on_change("active", update_umap)
    approximateMinimumSpanningTreeToggle.on_change("active", update_umap)
    # Attach the callback function to the MultiChoice's "value" property
    andFilterMultiChoice.on_change('value', update_umap)
    orFilterMultiChoice.on_change('value', update_umap)
    # Attach the callback to the checkbox
    highlightClusterCheckbox.on_change('active', updateCurrentCluster)

    gridDatasource.selected.on_change('indices', hover_callback)


    plot_figure.on_event(Tap, on_point_click)

    # Create a layout for the sliders and plot
    layout = column(mainLayout)

    if bokehShow:
        # Show the plot
        show(layout)
    else:
        # for Bokeh Server
        curdoc().add_root(layout)


def plotAndReturnSpikes(fluorescence_array, fps=30, number_consecutive_recordings=6):
    # interactive view

    # Calculate time values based on the frame rate (30 fps)
    n = len(fluorescence_array)
    time_values = np.arange(n) / fps

    # Plot intensity against time
    plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
    plt.plot(time_values, fluorescence_array, linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence Intensity')
    plt.title('Fluorescence Intensity vs. Time')
    plt.grid(True)

    # Add vertical lines at specific x-coordinates (assuming they have the same recording length)
    for i in range(1, number_consecutive_recordings):
        plt.axvline(x=((len(fluorescence_array) / fps) / number_consecutive_recordings) * i, color='black',
                    linestyle='--')

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
