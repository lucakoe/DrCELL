import random
import hdbscan
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import umap.plot
from bokeh.events import Tap
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, CustomJS, Slider, Select, Checkbox, \
    MultiChoice, ColorBar, TextInput, CustomAction, TapTool, CheckboxGroup, RadioGroup, Div, Button
from bokeh.palettes import Muted9
import io
import os

import util


def generate_diagnostic_plots(umap_object, data):
    mapper = umap_object.fit(data)
    umap.plot.diagnostic(mapper, diagnostic_type='pca')
    plt.show()
    umap.plot.diagnostic(mapper, diagnostic_type='vq')
    plt.show()
    umap.plot.diagnostic(mapper, diagnostic_type='local_dim')
    plt.show()
    umap.plot.diagnostic(mapper, diagnostic_type='neighborhood')
    plt.show()


def plot_bokeh(data_frames, datas, spike_plot_images_path, dump_files_paths, titles=None, bokeh_show=True,
               start_dropdown_data_option="all", output_file_path="./data/output/", data_variables=None,
               display_hover_variables=None, debug=False, experimental=False,
               color_palette=Muted9):
    if display_hover_variables is None:
        display_hover_variables = []
    display_hover_variables.insert(0,"Cluster")
    if data_variables is None:
        data_variables = []
    if titles is None:
        titles = ["all"]
    print("Loading Bokeh Plotting Interface")
    images_path = spike_plot_images_path

    for title in titles:
        data_frames[title] = data_frames[title].sample(frac=1, random_state=42)
        if debug: print(f"Dataframe {title}: \n{data_frames[title]}")
        data_frames[title]['OwnIndex'] = data_frames[title].index
        data_frames[title]['alpha'] = 1.0
        data_frames[title]['ColorMappingCategory'] = 1.0
        data_frames[title]['Cluster'] = -1

    datasource_df = pd.DataFrame.copy(data_frames[start_dropdown_data_option])
    update_cluster_toggle_df = pd.DataFrame.copy(datasource_df)
    # Create a ColumnDataSource
    datasource = ColumnDataSource(datasource_df)

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
    grid_start_pos = (0.0, 0.0)
    min_point = (datasource_df["x"].min(), datasource_df["y"].min())
    max_point = (datasource_df["x"].max(), datasource_df["y"].max())

    # Generate data points in the middle of each grid cell
    grid_datasource_df = util.generate_grid(min_point, max_point, center_point=grid_start_pos,
                                            grid_size_x=gridSizeX, grid_size_y=gridSizeY)

    grid_datasource_df['alpha'] = 0.1
    grid_datasource_df['gridSizeX'] = gridSizeX
    grid_datasource_df['gridSizeY'] = gridSizeY
    # grid_datasource_df = util.assign_points_to_grid(datasource_df, grid_datasource_df)
    grid_datasource = ColumnDataSource(grid_datasource_df)

    # Create a Bokeh plot
    grid_plot = plot_figure.rect('centerX', 'centerY', 'gridSizeX', 'gridSizeY',
                                 source=grid_datasource,
                                 fill_color="lightblue",
                                 line_color="black",
                                 line_alpha='alpha',
                                 fill_alpha='alpha')

    # Create a scatter plot
    scatter_plot = plot_figure.scatter(
        'x',
        'y',
        source=datasource,
        fill_color="blue",
        line_color="blue",
        line_alpha="alpha",
        fill_alpha="alpha",
        size=4,
        marker='circle'
    )

    # Custom Tools

    hover_variable_string = ""
    for variable in display_hover_variables:
        hover_variable_string += f"""<span style='font-size: 8px; color: #224499'>{variable}:</span>\n
                <span style='font-size: 8px'>@{variable}</span>\n"""

    scatter_plot_hover_tool = HoverTool(tooltips=f"""
        <div>
            {hover_variable_string}
        </div>

        <div>
            <img
                src="file://{images_path}""" + """/image_@{index}.png" height="100" alt="Image"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            />
    </div>
    """, renderers=[scatter_plot])
    # TODO finish grid hover tool
    grid_plot_hover_tool = HoverTool(tooltips="""
    <div>
        <span style='font-size: 8px; color: #224499'>Grid ID:</span>\n
        <span style='font-size: 8px'>@{gridID}</span>\n
        <span style='font-size: 8px; color: #224499'>Grid Point Indices:</span>\n
        <span style='font-size: 8px'>@{pointIndices}</span>\n
        <span style='font-size: 8px; color: #224499'>Grid Point Neurons:</span>\n
        <span style='font-size: 8px'>@{pointNeurons}</span>\n
        
    </div>
    <div>
        <img src="data:image/png;base64, @{image}" style="float: left; margin: 0px 15px 15px 0px; width: 100px; height: auto;">

    </div>
           
    </div>
    """, renderers=[grid_plot])

    # Add a HoverTool to display the Matplotlib plot when hovering over a data point
    plot_figure.add_tools(scatter_plot_hover_tool)
    plot_figure.add_tools(grid_plot_hover_tool)

    # Create an empty line Div for spacing
    blank_div = Div(text="<br>", width=400, height=5)

    # General

    general_title_div = Div(text="<h3>General: </h3>", width=400, height=20)

    options_select_data = titles
    select_data = Select(title="Data:", value=start_dropdown_data_option, options=options_select_data)

    options_filter_multi_choice_values = {}
    options_select_color = ["all", "Cluster"]

    for option in data_variables:
        options_select_color.append(option)
        for value in np.unique(data_frames[start_dropdown_data_option][option]):
            options_filter_multi_choice_values[f"{option} == {value}"] = (option, value)

    select_color = Select(title="Color:", value=options_select_color[0], options=options_select_color)

    options_filter_multi_choice = list(options_filter_multi_choice_values.keys())
    or_filter_multi_choice = MultiChoice(title="'OR' Filter:", value=[], options=options_filter_multi_choice, width=200)
    and_filter_multi_choice = MultiChoice(title="'AND' Filter:", value=[], options=options_filter_multi_choice,
                                          width=200)
    export_data_button = Button(label="Export Data")

    general_layout = column(general_title_div, blank_div, select_data, select_color, or_filter_multi_choice,
                            and_filter_multi_choice, export_data_button)

    # Hover Tool and Grid selection
    enable_grid_checkbox = Checkbox(label="Grid Enabled", active=False)
    grid_plot.visible = enable_grid_checkbox.active
    grid_size_x_text_input = TextInput(value="1.0", title="Grid Size X:", disabled=False)
    grid_size_y_text_input = TextInput(value="1.0", title="Grid Size Y:", disabled=False)
    grid_size_button = Button(label="Update")

    hover_tool_layout = column(enable_grid_checkbox, grid_size_x_text_input, grid_size_y_text_input, grid_size_button)

    # UMAP
    umap_title_div = Div(text="<h3>UMAP Parameters: </h3>", width=400, height=20)

    if experimental:
        n_neighbors_slider = Slider(title="n_neighbors", start=2, end=50, step=1, value=20)
        min_dist_slider = Slider(title="min_dist", start=0.00, end=1.0, step=0.01, value=0.0)
    else:
        n_neighbors_slider = Slider(title="n_neighbors", start=10, end=20, step=1, value=20)
        min_dist_slider = Slider(title="min_dist", start=0.00, end=0.10, step=0.01, value=0.0)

    umap_layout = column(umap_title_div, blank_div, n_neighbors_slider, min_dist_slider)

    # Cluster Parameters

    cluster_parameters_title_div = Div(text="<h3>Cluster Parameters: </h3>", width=400, height=20)
    update_clusters_toggle = Checkbox(label="Update Clusters (experimental)", active=True)
    min_cluster_size_slider = Slider(title="min_cluster_size", start=1, end=50, step=1, value=5, disabled=False)
    min_samples_slider = Slider(title="min_sample", start=1, end=10, step=1, value=1, disabled=False)
    options_cluster_selection_method = ['eom', 'leaf']
    cluster_selection_method_toggle = RadioGroup(labels=options_cluster_selection_method,
                                                 active=0)  # Set "eom" as the default
    cluster_selection_epsilon_slider = Slider(title="cluster_selection_epsilon", start=0.00, end=1.0, step=0.01,
                                              value=0.0)
    options_metric = ["euclidean", "manhattan", "correlation", "jaccard", "hamming", "chebyshev", "canberra",
                      "braycurtis"]
    select_metric = Select(title="metric:", value=options_metric[0], options=options_metric)

    allow_single_linkage_toggle = Checkbox(label="Allow Single-Linkage", active=False)
    approximate_minimum_spanning_tree_toggle = Checkbox(label="Approximate Minimum Spanning Tree", active=True)

    cluster_parameters_layout = column(cluster_parameters_title_div, blank_div, update_clusters_toggle,
                                       min_cluster_size_slider,
                                       min_samples_slider,
                                       cluster_selection_epsilon_slider,
                                       allow_single_linkage_toggle, approximate_minimum_spanning_tree_toggle,
                                       select_metric,
                                       cluster_selection_method_toggle)

    # Cluster Selection
    # TODO fix Cluster Selection bug
    cluster_selection_title_div = Div(text="<h3>Cluster Selection: </h3>", width=400, height=20)

    highlight_cluster_checkbox = Checkbox(label="Highlight Cluster", active=False)
    selected_cluster_text_input = TextInput(value="0", title="Cluster:", disabled=True)
    select_cluster_slider = Slider(title="cluster", start=-1,
                                   end=len(np.unique(data_frames[start_dropdown_data_option]['Cluster'])), step=1,
                                   value=0,
                                   disabled=True)
    cluster_selection_layout = column(cluster_selection_title_div, blank_div, highlight_cluster_checkbox,
                                      selected_cluster_text_input,
                                      select_cluster_slider)

    main_layout_title_div = Div(text="<h2>Cluster Exploration and Labeling Library: </h2>", width=800, height=20)

    main_layout_row = row(general_layout, column(umap_layout, cluster_selection_layout), cluster_parameters_layout)
    main_layout = column(main_layout_title_div, blank_div,
                         main_layout_row,
                         row(plot_figure, hover_tool_layout))

    current_cluster = 0

    # Callback function to update graph when sliders change
    def update_umap(attr, old, new):
        n_neighbors = n_neighbors_slider.value
        min_dist = min_dist_slider.value
        nonlocal datasource_df, update_cluster_toggle_df, current_cluster
        # Resets to initial state
        datasource_df = pd.DataFrame.copy(data_frames[select_data.value])
        # TODO fix Update Cluster
        if debug: print(datasource_df)
        if debug: print(update_cluster_toggle_df)
        if debug: print(update_cluster_toggle_df[update_cluster_toggle_df["Task"] == "1"])
        # Set the 'ID' column as the index in both DataFrames
        # datasource_df.set_index('OwnIndex', inplace=True)
        # update_cluster_toggle_df.set_index('OwnIndex', inplace=True)
        datasource_df.update(update_cluster_toggle_df["Cluster"])
        if debug: print(datasource_df[datasource_df["Task"] == "1"])
        # datasource_df.reset_index(inplace=True)

        umap_result = util.get_umap_out(datas[select_data.value],
                                        os.path.abspath(dump_files_paths[select_data.value]).replace("\\", '/'),
                                        n_neighbors=n_neighbors, min_dist=round(min_dist, 2))
        datasource_df['x'], datasource_df['y'] = umap_result[:, 0], umap_result[:, 1]
        # datasource.data.update({'x': umap_result[:, 0], 'y': umap_result[:, 1]})
        if len(or_filter_multi_choice.value) != 0:
            datasource_df = pd.DataFrame(columns=datasource_df.columns)
            for option in or_filter_multi_choice.value:
                initial_df = pd.DataFrame.copy(data_frames[select_data.value])
                # makes a dataframe with just the filtered entries and merges it with the other selected values
                filter_df = initial_df[
                    initial_df[options_filter_multi_choice_values[option][0]] ==
                    options_filter_multi_choice_values[option][
                        1]]
                datasource_df = pd.merge(datasource_df, filter_df, how='outer')

            datasource_df = datasource_df.drop_duplicates(keep="first")
            # shuffles Data order for plotting
            datasource_df = datasource_df.sample(frac=1, random_state=42)

        if len(and_filter_multi_choice.value) != 0:
            for option in and_filter_multi_choice.value:
                datasource_df = datasource_df[
                    datasource_df[options_filter_multi_choice_values[option][0]] ==
                    options_filter_multi_choice_values[option][
                        1]]

                if debug: print(type(datasource_df[options_filter_multi_choice_values[option][0]]))
                if debug: print(type(options_filter_multi_choice_values[option][1]))
                if debug: print(datasource_df[options_filter_multi_choice_values[option][0]])

        if debug: print(datasource_df)
        umap_result = datasource_df[['x', 'y']].values
        if update_clusters_toggle.active:
            clusters = []
            if len(umap_result) > 0:
                # Apply HDBSCAN clustering
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_slider.value,
                                            min_samples=min_samples_slider.value,
                                            allow_single_cluster=allow_single_linkage_toggle.active,
                                            approx_min_span_tree=approximate_minimum_spanning_tree_toggle.active,
                                            cluster_selection_method=options_cluster_selection_method[
                                                cluster_selection_method_toggle.active],
                                            metric=select_metric.value,
                                            cluster_selection_epsilon=cluster_selection_epsilon_slider.value)
                clusters = clusterer.fit_predict(umap_result)

            # Add cluster labels to your dataframe
            datasource_df['Cluster'] = clusters
        if debug: print(datasource_df)

        select_cluster_slider.end = len(np.unique(datasource_df['Cluster']))

        # Update the existing datasource
        datasource.data.update(ColumnDataSource(datasource_df).data)
        update_current_cluster(attr=None, old=None, new=None)
        update_grid(attr=None, old=None, new=None)
        datasource.data.update(datasource.data)
        update_category(attr=None, old=None, new=None)

    def update_current_cluster(attr, old, new):
        nonlocal current_cluster

        if highlight_cluster_checkbox.active:
            if current_cluster != select_cluster_slider.value:
                current_cluster = select_cluster_slider.value
                selected_cluster_text_input.value = str(current_cluster)
            elif current_cluster != int(selected_cluster_text_input.value):
                current_cluster = int(selected_cluster_text_input.value)
                select_cluster_slider.value = current_cluster

            select_cluster_slider.disabled = False
            selected_cluster_text_input.disabled = False
            print(
                f"Current Cluster: {current_cluster}, Current Cluster Size: {len(datasource_df[datasource_df['Cluster'] == current_cluster])}")
            for i, cluster in enumerate(datasource_df['Cluster']):  # Assuming cluster_data is a list of cluster labels
                if cluster == current_cluster:
                    datasource.data['alpha'][i] = 1  # Make points in the selected cluster fully visible
                else:
                    datasource.data['alpha'][i] = 0.05  # Make points in other clusters more transparent

        else:
            select_cluster_slider.disabled = True
            selected_cluster_text_input.disabled = True

            for i, cluster in enumerate(datasource_df['Cluster']):  # Assuming cluster_data is a list of cluster labels
                datasource.data['alpha'][i] = 1  # Make points in the selected cluster fully visible

        clustered_count = len(datasource_df[datasource_df['Cluster'] != -1])
        unclustered_count = len(datasource_df[datasource_df['Cluster'] == -1])

        # Check if the denominator (unclustered_count) is zero
        if unclustered_count == 0:
            ratio = "N/A"
        else:
            ratio = round(clustered_count / unclustered_count, 3)
        # Check if the denominator (unclustered_count) is zero
        if len(datasource_df) == 0:
            percentage = "N/A"
            cluster_number = 0
        else:
            percentage = round((clustered_count / len(datasource_df)) * 100, 2)
            cluster_number = len(np.unique(datasource_df['Cluster'])) - 1

        print(
            f"Data: {select_data.value}, 'AND' Filter: {and_filter_multi_choice.value}, 'OR' Filter: {or_filter_multi_choice.value}, Length: {len(datasource_df)}")
        print(
            f"Clusters: {cluster_number}, Clustered: {percentage}%, Clustered/Unclustered Ratio: {ratio}, Clustered: {clustered_count}, Unclustered: {unclustered_count}"
        )

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

    color_bar_initialized = False
    color_bar = ColorBar()

    def update_category(attr, old, new):
        nonlocal scatter_plot, color_bar, color_bar_initialized, color_palette
        if select_color.value == "all":
            scatter_plot.glyph.fill_color = "blue"
            scatter_plot.glyph.line_color = "blue"
            datasource.data.update(ColumnDataSource(datasource_df).data)
        else:
            unique_factors = np.unique(datasource_df[select_color.value])
            if select_color.value == "Cluster":
                unique_factors = unique_factors[unique_factors != -1]
            try:
                unique_factors = sorted(unique_factors)
                print(f"Color adjusted to and sorted by {select_color.value}")

            except TypeError:
                print(f"Color adjusted unsorted by {select_color.value}")
            new_factors = [str(x) for x in unique_factors]

            datasource_df['ColorMappingCategory'] = datasource_df[select_color.value].astype(str)
            datasource.data.update(ColumnDataSource(datasource_df).data)

            custom_color_palette = list(
                [color_palette[int(int(i * (len(color_palette) - 1) / len(unique_factors)))] for i in
                 range(len(unique_factors))])
            random.shuffle(custom_color_palette)
            color_mapping = CategoricalColorMapper(
                factors=new_factors,
                palette=custom_color_palette)

            scatter_plot.glyph.fill_color = {'field': 'ColorMappingCategory', 'transform': color_mapping}
            scatter_plot.glyph.line_color = {'field': 'ColorMappingCategory', 'transform': color_mapping}

            if not color_bar_initialized:
                # Create a color bar for the color mapper
                color_bar = ColorBar(title=select_color.value, color_mapper=color_mapping, location=(0, 0))
                # Add the color bar to the figure
                plot_figure.add_layout(color_bar, 'below')
                color_bar_initialized = True
            else:
                color_bar.color_mapper = color_mapping

            if select_color.value == "all":
                color_bar.visible = False
            else:
                color_bar.visible = True
                color_bar.title = select_color.value

    def update_grid_button():
        update_grid(attr=None, old=None, new=None)

    def update_cluster_toggle_function(attr, old, new):
        nonlocal update_cluster_toggle_df, datasource_df
        if update_clusters_toggle.active:
            update_cluster_toggle_df = pd.DataFrame.copy(datasource_df)
        update_umap(attr=None, old=None, new=None)

    def update_grid(attr, old, new):
        global grid_size_y, grid_size_x
        if enable_grid_checkbox.active:
            grid_size_x = float(grid_size_x_text_input.value)
            grid_size_y = float(grid_size_y_text_input.value)
            grid_start_pos = (0.0, 0.0)
            min_point = (datasource_df["x"].min(), datasource_df["y"].min())
            max_point = (datasource_df["x"].max(), datasource_df["y"].max())

            # Generate data points in the middle of each grid cell
            grid_datasource_df = util.generate_grid(min_point, max_point, center_point=grid_start_pos,
                                                    grid_size_x=grid_size_x, grid_size_y=grid_size_y)
            grid_datasource_df['gridSizeX'] = grid_size_x
            grid_datasource_df['gridSizeY'] = grid_size_y
            grid_datasource_df['alpha'] = 0.1
            grid_datasource_df = util.assign_points_to_grid(datasource_df, grid_datasource_df,
                                                            [('index', 'pointIndices'), ("Neuron", "pointNeurons")])

            grid_datasource.data.update(ColumnDataSource(grid_datasource_df).data)

        grid_plot.visible = enable_grid_checkbox.active

    def export_data():
        # Convert the DataFrame to a dictionary
        data_dict = {'df': datasource_df.to_dict("list")}
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_umap_cluster_output"

        # Save the dictionary to a MATLAB .mat file
        scipy.io.savemat(os.path.join(output_file_path, filename + ".mat"), data_dict)
        np.save(os.path.join(output_file_path, filename + ".npy"), datasource_df.to_numpy())
        print(f"Data has been saved to {filename}_umap_cluster_output.mat")
        print(f"Data has been saved to {filename}_umap_cluster_output.npy")

    def hover_callback(attr, old_index, new_index):
        if new_index:
            selected_data = grid_datasource.data
            selected_x = selected_data['x'][new_index[0]]
            selected_y = selected_data['y'][new_index[0]]
            selected_index = selected_data['index'][new_index[0]]
            print(f"Hovered over data point at x={selected_x}, y={selected_y}, index={selected_index}")

    # Attach the callback function to slider changes
    n_neighbors_slider.on_change('value_throttled', update_umap)
    min_dist_slider.on_change('value_throttled', update_umap)
    min_samples_slider.on_change('value_throttled', update_umap)
    cluster_selection_epsilon_slider.on_change('value_throttled', update_umap)
    select_cluster_slider.on_change('value_throttled', update_current_cluster)
    min_cluster_size_slider.on_change('value_throttled', update_umap)
    # Attach the callback function to the dropdown menu
    select_data.on_change('value', update_umap)
    select_metric.on_change('value', update_umap)
    select_color.on_change('value', update_category)
    selected_cluster_text_input.on_change('value', update_current_cluster)
    grid_size_button.on_click(update_grid_button)
    export_data_button.on_click(export_data)
    enable_grid_checkbox.on_change('active', update_grid)

    # Attach the callback function to the CheckboxGroup's active property
    cluster_selection_method_toggle.on_change("active", update_umap)
    allow_single_linkage_toggle.on_change("active", update_umap)
    approximate_minimum_spanning_tree_toggle.on_change("active", update_umap)
    update_clusters_toggle.on_change("active", update_cluster_toggle_function)
    # Attach the callback function to the MultiChoice's "value" property
    and_filter_multi_choice.on_change('value', update_umap)
    or_filter_multi_choice.on_change('value', update_umap)
    # Attach the callback to the checkbox
    highlight_cluster_checkbox.on_change('active', update_current_cluster)

    grid_datasource.selected.on_change('indices', hover_callback)

    plot_figure.on_event(Tap, on_point_click)

    # Create a layout for the sliders and plot
    layout = column(main_layout)
    update_umap(attr=None, old=None, new=None)

    if bokeh_show:
        # Show the plot
        show(layout)
    else:
        # for Bokeh Server
        curdoc().add_root(layout)
    print("Finished loading Bokeh Plotting Interface")


def plot_and_return_spikes(fluorescence_array, fps=30, number_consecutive_recordings=6):
    # Calculate time values based on the frame rate per second
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


def plot_and_save_spikes(neuron_number, dataframe, output_folder, fps=30, number_consecutive_recordings=6):
    # takes selected row (fluorescence data of one cell), makes it to an array and plots it
    plt = plot_and_return_spikes(dataframe.iloc[neuron_number].values, fps=fps,
                                 number_consecutive_recordings=number_consecutive_recordings)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80)
    buf.seek(0)
    # Ensure the output directory exists; create it if it doesn't
    os.makedirs(output_folder, exist_ok=True)

    # Save each image as a separate PNG file
    file_path = os.path.join(output_folder, f"image_{neuron_number}.png")
    with open(file_path, "wb") as file:
        file.write(buf.read())

    print(f"Saved image {neuron_number} to {file_path}")
    plt.close()
