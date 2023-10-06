import numpy as np
import matplotlib.pyplot as plt
import umap.plot
from bokeh.layouts import column
from umap import plot, UMAP
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, CustomJS, Slider
from bokeh.palettes import Spectral4
import io
import os


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


def plotBokeh(dataFrame, data):
    # Create a ColumnDataSource
    datasource = ColumnDataSource(dataFrame)
    # Define color mapping
    color_mapping = CategoricalColorMapper(factors=[str(x) for x in np.unique(dataFrame['Task'])],
                                           palette=Spectral4)

    # Create the Bokeh figure
    plot_figure = figure(
        title='UMAP projection of Cells',
        width=600,  # Use 'width' instead of 'plot_width'
        height=600,  # Use 'height' instead of 'plot_height'
        tools='pan, wheel_zoom, box_zoom,save, reset, help'
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

    # Define sliders for UMAP parameters
    n_neighbors_slider = Slider(title="n_neighbors", start=1, end=30, step=1, value=15)
    min_dist_slider = Slider(title="min_dist", start=0.01, end=1.0, step=0.01, value=0.1)

    # Callback function to update UMAP when sliders change
    def update_umap(attr, old, new):
        n_neighbors = n_neighbors_slider.value
        min_dist = min_dist_slider.value
        umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        umap_result = umap.fit_transform(data)
        datasource.data = {'x': umap_result[:, 0], 'y': umap_result[:, 1]}

    # Attach the callback function to slider changes
    n_neighbors_slider.on_change('value', update_umap)
    min_dist_slider.on_change('value', update_umap)

    # Create a layout for the sliders and plot
    layout = column(n_neighbors_slider, min_dist_slider, plot_figure)

    # Show the plot
    show(layout)


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
