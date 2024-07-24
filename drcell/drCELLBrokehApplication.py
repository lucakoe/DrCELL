import datetime
import json
import os
import random

import hdbscan
import numpy as np
import pandas as pd
import scipy
import sklearn.decomposition
from bokeh.events import Tap
from bokeh.io import show, curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, Div, Select, Button, MultiChoice, Checkbox, TextInput, Slider, \
    RadioGroup, ColorBar, CategoricalColorMapper
from bokeh.palettes import Muted9
from bokeh.plotting import figure
from matplotlib import pyplot as plt

import drcell.dimensionalReduction
import drcell.util
from drcell.dimensionalReduction import *
from drcell.server.ImageServer import ImageServer


def plot_bokeh(input_file_paths, reduction_functions=None,
               bokeh_show=True, output_path="./data/output/", debug=False, experimental=False,
               hover_image_generation_function=None,
               color_palette=Muted9, image_server_port=8000):
    reduction_functions_config_path = os.path.abspath('./drcell/config/reduction_functions_config.json')
    # TODO make Application class with those as attributes instead of global variables
    current_dataset = None
    current_pca_preprocessed_dataset = None
    image_server = ImageServer(image_server_port, current_dataset, current_pca_preprocessed_dataset)
    image_server.start_server()
    print("Starting DrCELL")
    if reduction_functions is None:
        reduction_functions = []

    reduction_functions = [drcell.dimensionalReduction.UMAPDRObject(),
                           drcell.dimensionalReduction.TSNEDRObject(),
                           drcell.dimensionalReduction.PHATEDRObject()] + reduction_functions
    default_reduction_function_name = reduction_functions[0].name
    # loads parameters and default values from config file
    if os.path.exists(reduction_functions_config_path):
        with open(reduction_functions_config_path, 'r') as json_file:
            reduction_function_config = json.load(json_file)
        for reduction_function in reduction_functions:
            if reduction_function.name in reduction_function_config.keys():
                print(
                    f"Loading {reduction_function.name} config from {os.path.basename(reduction_functions_config_path)}")
                reduction_function.change_params(reduction_function_config[reduction_function.name])
    reduction_functions = {dr_object.get_name(): dr_object for dr_object in reduction_functions}

    datas = {}
    legend_dfs = {}
    configs = {}
    file_folder_paths = {}
    data_frames = {}

    for file in input_file_paths:
        title = os.path.splitext(os.path.basename(file))[0]
        file_folder_paths[title] = drcell.util.generalUtil.create_file_folder_structure(output_path, title)

        datas[title], legend_dfs[title], configs[title] = drcell.util.drCELLFileUtil.load_dr_cell_h5(file)
        # TODO check if its alright for datas to be a df. otherwise convert it here to np array
        datas[title] = datas[title].to_numpy()
        if ("data_variables" not in configs[title]) or configs[title]["data_variables"] is None:
            configs[title]["data_variables"] = []
        else:
            configs[title]["data_variables"] = list(configs[title]["data_variables"])
        if ("display_hover_variables" not in configs[title]) or configs[title][
            "display_hover_variables"] is None:
            configs[title]["display_hover_variables"] = []
        else:
            configs[title]["display_hover_variables"] = list(configs[title]["display_hover_variables"])
        configs[title]["display_hover_variables"].insert(0, "Cluster")
        if ("recording_type" not in configs[title]) or configs[title]["recording_type"] is None:
            configs[title]["recording_type"] = "None"

        temp_umap_out = reduction_functions[default_reduction_function_name].get_dimensional_reduction_out(
            datas[title], dump_folder_path=file_folder_paths[title],
            reduction_params=reduction_functions[default_reduction_function_name].get_default_params(),
            pca_preprocessing=False)

        if debug: print('Umap vals: ' + str(temp_umap_out.shape))

        # Apply HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1)
        clusters = clusterer.fit_predict(temp_umap_out)
        current_clusterer = clusterer

        data_frames[title] = pd.DataFrame(temp_umap_out, columns=['x', 'y'])
        # creates an index for merging
        data_frames[title].index = range(len(data_frames[title]))
        legend_dfs[title].index = range(len(legend_dfs[title]))
        data_frames[title] = data_frames[title].merge(legend_dfs[title], left_index=True, right_index=True)
        # Add cluster labels to your dataframe
        data_frames[title]['Cluster'] = clusters

        data_frames[title] = data_frames[title].sample(frac=1, random_state=42)
        if debug: print(f"Dataframe {title}: \n{data_frames[title]}")
        data_frames[title]['pdIndex'] = data_frames[title].index
        data_frames[title]['alpha'] = 1.0
        data_frames[title]['ColorMappingCategory'] = 1.0
        data_frames[title]['Cluster'] = -1
        data_frames[title]['recordingType'] = configs[title]["recording_type"]

    print("Loading Bokeh Plotting Interface")

    datasource_df = pd.DataFrame.copy(data_frames[list(datas.keys())[0]])
    update_cluster_toggle_df = pd.DataFrame.copy(datasource_df)
    # Create a ColumnDataSource
    datasource = ColumnDataSource(datasource_df)

    # Create the Bokeh figure
    plot_figure = figure(
        title='Graph',
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
    grid_datasource_df = drcell.util.generalUtil.generate_grid(min_point, max_point, center_point=grid_start_pos,
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
    scatter_plot_hover_tool = HoverTool(tooltips="""<span style='font-size: 8px'>init</span>\n""",
                                        renderers=[scatter_plot])
    scatter_plot_pca_preprocessing_hover_tool = HoverTool(tooltips="""<span style='font-size: 8px'>init</span>\n""",
                                                          renderers=[scatter_plot])
    grid_plot_hover_tool = HoverTool(name="Grid Median Hovertool",
                                     tooltips="""<span style='font-size: 8px'>init</span>\n""", renderers=[grid_plot])

    # Add a HoverTool to display the Matplotlib plot when hovering over a data point
    plot_figure.add_tools(scatter_plot_hover_tool)
    plot_figure.add_tools(grid_plot_hover_tool)
    plot_figure.add_tools(scatter_plot_pca_preprocessing_hover_tool)

    # Create an empty line Div for spacing
    blank_div = Div(text="<br>", width=400, height=5)

    # General

    general_title_div = Div(text="<h3>General: </h3>", width=400, height=20)

    select_data = Select(title="Data:", value=list(datas.keys())[0], options=list(datas.keys()))

    options_filter_multi_choice_values = {}
    options_select_color = ["all", "Cluster"]

    for option in configs[list(datas.keys())[0]]["data_variables"]:
        options_select_color.append(option)
        for value in np.unique(data_frames[list(datas.keys())[0]][option]):
            options_filter_multi_choice_values[f"{option} == {value}"] = (option, value)

    select_color = Select(title="Color:", value=options_select_color[0], options=options_select_color)
    randomize_colors_button = Button(label="Randomize Colors")

    options_filter_multi_choice = list(options_filter_multi_choice_values.keys())
    or_filter_multi_choice = MultiChoice(title="'OR' Filter:", value=[], options=options_filter_multi_choice, width=200)
    and_filter_multi_choice = MultiChoice(title="'AND' Filter:", value=[], options=options_filter_multi_choice,
                                          width=200)
    export_data_button = Button(label="Export Data")
    export_only_selection_toggle = Checkbox(label="Export only selection", active=True)
    options_export_sort_category = datasource_df.columns.tolist()
    select_export_sort_category = Select(title="Sort export by", value="Cluster", options=options_export_sort_category)

    general_layout = column(general_title_div, blank_div, select_data, select_color, randomize_colors_button,
                            or_filter_multi_choice,
                            and_filter_multi_choice, row(export_data_button, export_only_selection_toggle),
                            select_export_sort_category)

    # Stats
    stats_title_div = Div(text="<h3>Statistics: </h3>", width=400, height=20)
    stats_div = Div(text="<h2>stat init</h2>", width=400, height=100)
    stats_layout = column(stats_title_div, blank_div, stats_div)

    # Hover Tool and Grid selection
    grid_title_div = Div(text="<h3>Grid Settings: </h3>", width=400, height=20)
    enable_grid_checkbox = Checkbox(label="Grid Enabled", active=False)
    grid_plot.visible = enable_grid_checkbox.active
    grid_size_x_text_input = TextInput(value="1.0", title="Grid Size X:", disabled=False)
    grid_size_y_text_input = TextInput(value="1.0", title="Grid Size Y:", disabled=False)
    grid_size_button = Button(label="Update")

    hover_tool_layout = column(grid_title_div, blank_div, enable_grid_checkbox, grid_size_x_text_input,
                               grid_size_y_text_input, grid_size_button)

    # PCA Preprocessing

    pca_preprocessing_title_div = Div(text="<h3>PCA Preprocessing: </h3>", width=400, height=20)

    enable_pca_checkbox = Checkbox(label="Enable PCA Preprocessing", active=False)
    select_pca_dimensions_slider = Slider(title="PCA n_components", start=1,
                                          end=min(datas[list(datas.keys())[0]].shape[0],
                                                  datas[list(datas.keys())[0]].shape[1]), step=1,
                                          value=2,
                                          disabled=True)
    pca_diagnostic_plot_button = Button(label="PCA Diagnostic Plot")
    pca_preprocessing_layout = column(pca_preprocessing_title_div, blank_div, enable_pca_checkbox,
                                      select_pca_dimensions_slider, pca_diagnostic_plot_button)

    # Dimensional Reduction
    dimensional_reduction_title_div = Div(text="<h3>Dimensional Reduction: </h3>", width=400, height=20)
    buffer_parameters_button = Button(label="Buffer Dimensional Reduction in Parameter Range")
    buffer_parameters_status = Div(text=" ", width=400, height=20)
    options_select_dimensional_reduction = ["None"]
    options_select_dimensional_reduction.extend(list(reduction_functions.keys()))
    select_dimensional_reduction = Select(value="UMAP", options=options_select_dimensional_reduction)
    dimensional_reduction_parameter_layouts = column()
    reduction_functions_layouts = {}
    reduction_functions_widgets = {}
    # adds all the parameters from the reduction function as widgets to the interface.
    # Numeric parameters get added as Sliders, bool as checkboxes, select as Select and Constants get added later on.
    for reduction_function_name in reduction_functions.keys():
        reduction_functions_layouts[reduction_function_name] = column()
        reduction_functions_widgets[reduction_function_name] = {}
        reduction_function_params_dict = reduction_functions[reduction_function_name].get_DR_parameters_dict()

        for diagnostic_function_name in reduction_functions[reduction_function_name].list_diagnostic_functions_names():
            reduction_functions_widgets[reduction_function_name][diagnostic_function_name] = Button(
                label=diagnostic_function_name)
            reduction_functions_layouts[reduction_function_name].children.append(
                reduction_functions_widgets[reduction_function_name][diagnostic_function_name])

        for numerical_parameter in reduction_function_params_dict["numerical_parameters"].keys():
            parameter_range = reduction_function_params_dict["numerical_parameters"][numerical_parameter]
            reduction_functions_widgets[reduction_function_name][numerical_parameter] = Slider(
                title=numerical_parameter,
                **parameter_range)
            reduction_functions_layouts[reduction_function_name].children.append(
                reduction_functions_widgets[reduction_function_name][numerical_parameter])

        for bool_parameter in reduction_function_params_dict["bool_parameters"].keys():
            reduction_functions_widgets[reduction_function_name][bool_parameter] = Checkbox(label=bool_parameter,
                                                                                            active=
                                                                                            reduction_function_params_dict[
                                                                                                "bool_parameters"][
                                                                                                bool_parameter])
            reduction_functions_layouts[reduction_function_name].children.append(
                reduction_functions_widgets[reduction_function_name][bool_parameter])

        for nominal_parameter in reduction_function_params_dict["nominal_parameters"].keys():
            nominal_parameters_options = \
                reduction_function_params_dict["nominal_parameters"][nominal_parameter]["options"]
            nominal_parameters_default_option = \
                reduction_function_params_dict["nominal_parameters"][nominal_parameter]["default_option"]
            reduction_functions_widgets[reduction_function_name][nominal_parameter] = Select(
                value=nominal_parameters_default_option, options=nominal_parameters_options)
            reduction_functions_layouts[reduction_function_name].children.append(
                reduction_functions_widgets[reduction_function_name][nominal_parameter])

        dimensional_reduction_parameter_layouts.children.append(reduction_functions_layouts[reduction_function_name])

    dimensional_reduction_layout = column(dimensional_reduction_title_div, blank_div, select_dimensional_reduction,
                                          buffer_parameters_button, buffer_parameters_status,
                                          dimensional_reduction_parameter_layouts)

    # Cluster Parameters

    cluster_parameters_title_div = Div(text="<h3>Cluster Parameters: </h3>", width=400, height=20)
    update_clusters_toggle = Checkbox(label="Update Clusters (experimental)", active=True)
    hdbscan_diagnostic_plot_button = Button(label="HDBSCAN Diagnostic Plot")
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
                                       hdbscan_diagnostic_plot_button,
                                       min_cluster_size_slider,
                                       min_samples_slider,
                                       cluster_selection_epsilon_slider,
                                       allow_single_linkage_toggle, approximate_minimum_spanning_tree_toggle,
                                       select_metric,
                                       cluster_selection_method_toggle)

    # Cluster Selection
    # TODO fix Cluster Selection bug
    cluster_selection_title_div = Div(text="<h3>Highlight Cluster: </h3>", width=400, height=20)

    highlight_cluster_checkbox = Checkbox(label="Highlight Cluster", active=False)
    selected_cluster_text_input = TextInput(value="0", title="Cluster:", disabled=True)
    select_cluster_slider = Slider(title="Cluster", start=-1,
                                   end=len(np.unique(data_frames[list(datas.keys())[0]]['Cluster'])), step=1,
                                   value=0,
                                   disabled=True)
    cluster_selection_layout = column(cluster_selection_title_div, blank_div, highlight_cluster_checkbox,
                                      selected_cluster_text_input,
                                      select_cluster_slider)

    main_layout_title_div = Div(text="<h2>Dimensional reduction Cluster Exploration and Labeling Library: </h2>",
                                width=800, height=20)

    main_layout_row = row(general_layout,
                          column(pca_preprocessing_layout, dimensional_reduction_layout),
                          cluster_parameters_layout)
    main_layout = column(main_layout_title_div, blank_div,
                         main_layout_row,
                         row(plot_figure, column(stats_layout, cluster_selection_layout, hover_tool_layout)))

    current_cluster = 0

    def get_current_dimension_reduction_parameters():
        out = {}
        reduction_function_name = select_dimensional_reduction.value
        if reduction_function_name != 'None':
            for numeric_parameter in reduction_functions[reduction_function_name].get_DR_parameters_dict()[
                "numerical_parameters"].keys():
                value = reduction_functions_widgets[reduction_function_name][numeric_parameter].value
                if type(value) == float:
                    # rounds the value to the same amount of numbers behind the decimal point as the step of the slider.
                    # this is to prevent weird behavior with floats when buffering values
                    round(value, drcell.util.generalUtil.get_decimal_places(
                        reduction_functions_widgets[reduction_function_name][numeric_parameter].step))
                out[numeric_parameter] = value

            for bool_parameter in reduction_functions[reduction_function_name].get_DR_parameters_dict()[
                "bool_parameters"].keys():
                out[bool_parameter] = reduction_functions_widgets[reduction_function_name][bool_parameter].active

            for nominal_parameter in reduction_functions[reduction_function_name].get_DR_parameters_dict()[
                "nominal_parameters"].keys():
                out[nominal_parameter] = reduction_functions_widgets[reduction_function_name][nominal_parameter].value

            for constant_parameter in reduction_functions[reduction_function_name].get_DR_parameters_dict()[
                "constant_parameters"].keys():
                out[constant_parameter] = \
                    reduction_functions[reduction_function_name].get_DR_parameters_dict()["constant_parameters"][
                        constant_parameter]

        return out

    # Callback function to update graph when sliders change
    current_select_value = None

    def update_graph(attr, old, new):
        nonlocal current_select_value, datasource_df, update_cluster_toggle_df, current_cluster, options_select_color, select_color, datasource
        global current_pca_preprocessed_dataset
        # Resets to initial state
        datasource_df = pd.DataFrame.copy(data_frames[select_data.value])
        select_pca_dimensions_slider.end = min(datas[select_data.value].shape[0],
                                               datas[select_data.value].shape[1])

        if not current_select_value == select_data.value:
            # reset option to activate update cluster checkbox again
            update_clusters_toggle.active = True
            # updates the color selection according to the new dataset
            options_select_color = ["all", "Cluster"]
            for option in configs[select_data.value]["data_variables"]:
                options_select_color.append(option)
                for value in np.unique(data_frames[select_data.value][option]):
                    options_filter_multi_choice_values[f"{option} == {value}"] = (option, value)
            select_color.value = options_select_color[0]
            select_color.options = options_select_color

            hover_variable_string = ""
            for variable in configs[select_data.value]["display_hover_variables"]:
                hover_variable_string += f"""<span style='font-size: 8px; color: #224499'>{variable}:</span>\n
                        <span style='font-size: 8px'>@{variable}</span>\n"""

            scatter_plot_hover_tool.tooltips = f"""
                <div>
                    {hover_variable_string}
                </div>
                <div>
                    <img
                        src="http://localhost:""" + str(image_server_port) + """/?generate=""" + """@{pdIndex} &recording-type=@{recordingType}" height="100" alt="Image"
                        style="float: left; margin: 0px 15px 15px 0px;"
                        border="1"
                    />
                </div>
            """

            scatter_plot_pca_preprocessing_hover_tool.tooltips = f"""
                <div>
                    {hover_variable_string}
                </div>
                <div>
                    <img
                        src="http://localhost:""" + str(image_server_port) + """/?generate=""" + """@{pdIndex}&pca-preprocessing=True" height="100" alt="Image"
                        style="float: left; margin: 0px 15px 15px 0px;"
                        border="1"
                    />
                </div>
            """

            grid_plot_hover_tool.tooltips = """
                <div>
                    <span style='font-size: 8px; color: #224499'>Grid ID:</span>\n
                    <span style='font-size: 8px'>@{gridID}</span>\n 
                </div>
                <img
                    src="http://localhost:""" + str(image_server_port) + """/?generate=""" + """@{pointIndices}&extend-plot=True" height="100" alt="Image"
                        style="float: left; margin: 0px 15px 15px 0px;"
                        border="1"
                    />
                </div>
                """

        current_select_value = select_data.value

        # TODO fix Update Cluster
        if debug: print(datasource_df)
        if debug: print(update_cluster_toggle_df)
        if debug: print(update_cluster_toggle_df[update_cluster_toggle_df["Task"] == "1"])
        # Set the 'ID' column as the index in both DataFrames
        # datasource_df.set_index('pdIndex', inplace=True)
        # update_cluster_toggle_df.set_index('pdIndex', inplace=True)
        # TODO fix index problem: update_cluster_df has new index starting from 0 --> use pdIndex instead
        update_cluster_toggle_df.set_index(update_cluster_toggle_df["pdIndex"])
        datasource_df.set_index(datasource_df["pdIndex"])
        # IDEA: merge the two dataframes and just keep first and potentially updated version
        # datasource_df=pd.merge(update_cluster_toggle_df,datasource_df,on="pdIndex")
        # datasource_df.drop_duplicates(keep="first")
        datasource_df["Cluster"].update(update_cluster_toggle_df["Cluster"])
        if debug: print(datasource_df[datasource_df["Task"] == "1"])
        # datasource_df.reset_index(inplace=True)
        current_data = datas[select_data.value]
        print(get_current_dimension_reduction_parameters())

        if enable_pca_checkbox.active or select_dimensional_reduction.value == "None":
            if scatter_plot_pca_preprocessing_hover_tool not in plot_figure.tools:
                plot_figure.add_tools(scatter_plot_pca_preprocessing_hover_tool)
                pca_diagnostic_plot_button.disabled = False
            # this is to prevent the pca settings to be used if there is no dimensional reduction selected, so the data still gets reduced to 2 dimensions
            if select_dimensional_reduction.value == "None":
                enable_pca_checkbox.active = True
                enable_pca_checkbox.disabled = True
                select_pca_dimensions_slider.value = 2
                select_pca_dimensions_slider.disabled = True
                buffer_parameters_button.disabled = True
                current_pca_preprocessed_dataset = DimensionalReductionObject.apply_pca_preprocessing(
                    current_data, n_components=int(
                        select_pca_dimensions_slider.value))
                reduction_function_output = current_pca_preprocessed_dataset
            else:
                enable_pca_checkbox.disabled = False
                select_pca_dimensions_slider.disabled = False
                buffer_parameters_button.disabled = False
                reduction_function_output = reduction_functions[
                    select_dimensional_reduction.value].get_dimensional_reduction_out(
                    current_data,
                    dump_folder_path=file_folder_paths[select_data.value],
                    reduction_params=get_current_dimension_reduction_parameters(),
                    pca_preprocessing=True,
                    pca_n_components=int(select_pca_dimensions_slider.value))
                current_pca_preprocessed_dataset = DimensionalReductionObject.apply_pca_preprocessing(
                    current_data, n_components=int(
                        select_pca_dimensions_slider.value))



        else:
            select_pca_dimensions_slider.disabled = True
            pca_diagnostic_plot_button.disabled = True
            buffer_parameters_button.disabled = False
            if scatter_plot_pca_preprocessing_hover_tool in plot_figure.tools:
                plot_figure.remove_tools(scatter_plot_pca_preprocessing_hover_tool)
            current_pca_preprocessed_dataset = None
            reduction_function_output = reduction_functions[
                select_dimensional_reduction.value].get_dimensional_reduction_out(
                current_data,
                dump_folder_path=file_folder_paths[select_data.value],
                reduction_params=get_current_dimension_reduction_parameters(),
                pca_preprocessing=False)

        datasource_df['x'], datasource_df['y'] = reduction_function_output[:, 0], reduction_function_output[:, 1]
        data_frames[select_data.value]['x'], data_frames[select_data.value]['y'] = reduction_function_output[:,
                                                                                   0], reduction_function_output[:, 1]
        # datasource.data.update({'x': umap_result[:, 0], 'y': umap_result[:, 1]})
        initial_df = pd.DataFrame.copy(datasource_df)
        if len(or_filter_multi_choice.value) != 0:
            datasource_df = pd.DataFrame(columns=datasource_df.columns)
            for option in or_filter_multi_choice.value:
                current_df = pd.DataFrame.copy(initial_df)
                # makes a dataframe with just the filtered entries and merges it with the other selected values
                filter_df = current_df[
                    current_df[options_filter_multi_choice_values[option][0]] ==
                    options_filter_multi_choice_values[option][
                        1]]
                datasource_df = pd.merge(datasource_df, filter_df, how='outer')

            datasource_df = datasource_df.drop_duplicates(keep="first")

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
        reduction_function_output = datasource_df[['x', 'y']].values
        if update_clusters_toggle.active:
            clusters = []
            if len(reduction_function_output) > 0:
                # Apply HDBSCAN clustering
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_slider.value,
                                            min_samples=min_samples_slider.value,
                                            allow_single_cluster=allow_single_linkage_toggle.active,
                                            approx_min_span_tree=approximate_minimum_spanning_tree_toggle.active,
                                            cluster_selection_method=options_cluster_selection_method[
                                                cluster_selection_method_toggle.active],
                                            metric=select_metric.value,
                                            cluster_selection_epsilon=cluster_selection_epsilon_slider.value)
                clusters = clusterer.fit_predict(reduction_function_output)
                nonlocal current_clusterer
                current_clusterer = clusterer

            # Add cluster labels to your dataframe
            datasource_df['Cluster'] = clusters
        if debug: print(datasource_df)
        # shuffles Data order for plotting
        datasource_df = datasource_df.sample(frac=1, random_state=42)
        if update_clusters_toggle.active:
            update_cluster_toggle_df = pd.DataFrame.copy(datasource_df, True)
        if debug: print(datasource_df)
        select_cluster_slider.end = len(np.unique(datasource_df['Cluster']))

        # Update the existing datasource
        datasource.data.update(ColumnDataSource(datasource_df).data)
        update_current_cluster(attr=None, old=None, new=None)
        update_grid(attr=None, old=None, new=None)
        datasource.data.update(datasource.data)
        update_category(attr=None, old=None, new=None)
        update_current_dataset(current_data, current_pca_preprocessed_dataset, image_server=image_server)

    def update_current_dataset(dataset, pca_preprocessed_dataset, image_server=None):
        nonlocal current_dataset, current_pca_preprocessed_dataset
        current_dataset = dataset
        current_pca_preprocessed_dataset= pca_preprocessed_dataset
        if not image_server is None:
            image_server.update_dataset(current_dataset)
            image_server.update_pca_preprocessed_dataset(current_pca_preprocessed_dataset)

    def update_dimensional_reduction(attr, old, new):
        for reduction_functions_layout_name in list(reduction_functions_layouts.keys()):
            if select_dimensional_reduction.value == reduction_functions_layout_name:
                reduction_functions_layouts[reduction_functions_layout_name].visible = True
            else:
                reduction_functions_layouts[reduction_functions_layout_name].visible = False

        update_graph(attr=None, old=None, new=None)

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

        update_stats()

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

    def update_category_button():
        update_category(attr=None, old=None, new=None)

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

    def update_grid(attr, old, new):
        global grid_size_y, grid_size_x
        if enable_grid_checkbox.active:
            grid_size_x = float(grid_size_x_text_input.value)
            grid_size_y = float(grid_size_y_text_input.value)
            grid_start_pos = (0.0, 0.0)
            min_point = (datasource_df["x"].min(), datasource_df["y"].min())
            max_point = (datasource_df["x"].max(), datasource_df["y"].max())

            # Generate data points in the middle of each grid cell
            grid_datasource_df = drcell.util.generalUtil.generate_grid(min_point, max_point,
                                                                       center_point=grid_start_pos,
                                                                       grid_size_x=grid_size_x, grid_size_y=grid_size_y)
            grid_datasource_df['gridSizeX'] = grid_size_x
            grid_datasource_df['gridSizeY'] = grid_size_y
            grid_datasource_df['alpha'] = 0.1
            grid_datasource_df = drcell.util.generalUtil.assign_points_to_grid(datasource_df, grid_datasource_df,
                                                                               [('index', 'pointIndices'),
                                                                                ("Neuron", "pointNeurons")])

            grid_datasource.data.update(ColumnDataSource(grid_datasource_df).data)

        grid_plot.visible = enable_grid_checkbox.active

    def update_stats():
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

        if current_dataset is not None:
            dimensions_input_data_length = current_dataset.shape[1]
        else:
            dimensions_input_data_length = "N/A"
        print(
            f"Data: {select_data.value}, 'AND' Filter: {and_filter_multi_choice.value}, 'OR' Filter: {or_filter_multi_choice.value}, Datapoints: {len(datasource_df)}, Dimensions input data: {dimensions_input_data_length}")
        print(
            f"Clusters: {cluster_number}, Clustered: {percentage}%, Clustered/Unclustered Ratio: {ratio}, Clustered: {clustered_count}, Unclustered: {unclustered_count}"
        )
        stats_div.text = f"Data Points: {len(datasource_df)} <br> Dimensions input data: {dimensions_input_data_length} <br> Clusters: {cluster_number} <br> Clustered: {percentage}% <br> Clustered/Unclustered Ratio: {ratio} <br> Clustered: {clustered_count} <br> Unclustered: {unclustered_count} "

    def export_data():
        file_folder_output_path = os.path.join(file_folder_paths[select_data.value], "output")
        export_df = datasource_df
        if not export_only_selection_toggle.active:
            export_df = pd.DataFrame.copy(data_frames[select_data.value])

            export_df.update(datasource_df)

        export_df = export_df.sort_values(by=select_export_sort_category.value)

        # Convert the DataFrame to a dictionary
        data_dict = {'df': export_df.to_dict("list")}
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_umap_cluster_output"

        # Save the dictionary to a MATLAB .mat file
        scipy.io.savemat(os.path.join(file_folder_output_path, filename + ".mat"), data_dict)
        np.save(os.path.join(file_folder_output_path, filename + ".npy"), export_df.to_numpy())
        print(f"Data has been saved to {filename}_umap_cluster_output.mat")
        print(f"Data has been saved to {filename}_umap_cluster_output.npy")

    def hover_callback(attr, old_index, new_index):
        # plot_and_return_spike_images_b64()
        if new_index:
            selected_data = grid_datasource.data
            selected_x = selected_data['x'][new_index[0]]
            selected_y = selected_data['y'][new_index[0]]
            selected_index = selected_data['index'][new_index[0]]
            print(f"Hovered over data point at x={selected_x}, y={selected_y}, index={selected_index}")

    def pca_diagnostic_plot_button_callback():
        pca_operator = sklearn.decomposition.PCA(n_components=int(select_pca_dimensions_slider.value))
        pca = pca_operator.fit_transform(datas[select_data.value])
        diagnostic_data = pca_operator.explained_variance_ratio_
        DimensionalReductionObject.return_pca_diagnostic_plot(diagnostic_data).show()

    def hdbscan_diagnostic_plot_button_callback():
        nonlocal current_clusterer
        current_clusterer.condensed_tree_.plot()
        plt.show()
        current_clusterer.single_linkage_tree_.plot()
        plt.show()
        # clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
        #                                       edge_alpha=0.6,
        #                                       node_size=80,
        #                                       edge_linewidth=2)
        # plt.show()

    def diagnostic_plot_button_callback(diagnostic_function_name: str):
        nonlocal current_dataset
        parm = get_current_dimension_reduction_parameters()
        parm["data"] = current_dataset
        # returns the corresponding diagnostic function and executes it with data as parameter
        reduction_functions[select_dimensional_reduction.value].get_diagnostic_function(diagnostic_function_name)(
            **parm)

    def buffer_parameters():
        nonlocal buffer_parameters_status
        # TODO fix buffer status
        if select_dimensional_reduction.value != "None":
            buffer_parameters_status.text = "Start buffering"
            print("Start buffering")

            reduction_functions[select_dimensional_reduction.value].buffer_DR_in_paramter_range(datas[select_data.value],
                                                                                                file_folder_paths[
                                                                                                    select_data.value],
                                                                                                pca_preprocessing=enable_pca_checkbox.active,
                                                                                                pca_n_components=select_pca_dimensions_slider.value)
            buffer_parameters_status.visible = False
            print("Finished buffering")

    # Attach the callback function to Interface widgets

    # General
    select_data.on_change('value', update_graph)
    select_color.on_change('value', update_category)
    randomize_colors_button.on_click(update_category_button)
    or_filter_multi_choice.on_change('value', update_graph)
    and_filter_multi_choice.on_change('value', update_graph)
    export_data_button.on_click(export_data)

    # PCA Preprocessing

    enable_pca_checkbox.on_change('active', update_graph)
    select_pca_dimensions_slider.on_change('value_throttled', update_graph)
    pca_diagnostic_plot_button.on_click(pca_diagnostic_plot_button_callback)

    # Dimensional Reduction
    select_dimensional_reduction.on_change('value', update_dimensional_reduction)
    buffer_parameters_button.on_click(buffer_parameters)
    # assign the callback function for every parameter
    for reduction_function_name in reduction_functions.keys():
        for diagnostic_function_name in reduction_functions[reduction_function_name].list_diagnostic_functions_names():
            # assigns the corresponding diagnostic function button to a general callback function, that executes
            # the function based on the label name (that corresponds to the name of the diagnostic function)
            # and the currently selected reduction function
            reduction_functions_widgets[reduction_function_name][diagnostic_function_name].on_click(
                lambda: diagnostic_plot_button_callback(diagnostic_function_name))
        for numerical_parameter in reduction_functions[reduction_function_name].get_DR_parameters_dict()[
            "numerical_parameters"].keys():
            reduction_functions_widgets[reduction_function_name][numerical_parameter].on_change('value_throttled',
                                                                                                update_graph)
        for bool_parameter in reduction_functions[reduction_function_name].get_DR_parameters_dict()[
            "bool_parameters"].keys():
            reduction_functions_widgets[reduction_function_name][bool_parameter].on_change("active", update_graph)
        for nominal_parameter in reduction_functions[reduction_function_name].get_DR_parameters_dict()[
            "nominal_parameters"].keys():
            reduction_functions_widgets[reduction_function_name][nominal_parameter].on_change('value', update_graph)

    # Cluster Selection
    highlight_cluster_checkbox.on_change('active', update_current_cluster)
    selected_cluster_text_input.on_change('value', update_current_cluster)
    select_cluster_slider.on_change('value_throttled', update_current_cluster)

    # Cluster Parameters
    hdbscan_diagnostic_plot_button.on_click(hdbscan_diagnostic_plot_button_callback)
    update_clusters_toggle.on_change("active", update_graph)
    min_cluster_size_slider.on_change('value_throttled', update_graph)
    min_samples_slider.on_change('value_throttled', update_graph)
    cluster_selection_epsilon_slider.on_change('value_throttled', update_graph)
    allow_single_linkage_toggle.on_change("active", update_graph)
    approximate_minimum_spanning_tree_toggle.on_change("active", update_graph)
    select_metric.on_change('value', update_graph)
    cluster_selection_method_toggle.on_change("active", update_graph)

    # Hover Tool and Grid selection
    enable_grid_checkbox.on_change('active', update_grid)
    grid_size_button.on_click(update_grid_button)
    grid_datasource.selected.on_change('indices', hover_callback)
    plot_figure.on_event(Tap, on_point_click)

    # Create a layout for the sliders and plot
    layout = column(main_layout)
    update_dimensional_reduction(attr=None, old=None, new=None)

    if bokeh_show:
        # Show the plot
        show(layout)
    else:
        # for Bokeh Server
        curdoc().add_root(layout)
    update_stats()
    update_graph(attr=None, old=None, new=None)
    print("Finished loading Bokeh Plotting Interface")
