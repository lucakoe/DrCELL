import os
import socket
from decimal import Decimal

import pandas as pd
from scipy import io as sio


def print_mat_file(file_path):
    try:
        # Load the MATLAB file
        data = sio.loadmat(file_path)

        # Print the contents of the MATLAB file
        print("MATLAB File Contents:")
        for variable_name in data:
            print(f"Variable: {variable_name}")
            print(data[variable_name])
            print("\n")
    except Exception as e:
        print(f"Error: {e}")


def get_decimal_places(number):
    return len(str(number).split('.')[1])

    decimal_places = Decimal(str(number)).as_tuple().exponent
    return max(0, -decimal_places)


def generate_grid(min_point, max_point, center_point=(0.0, 0.0), grid_size_x=1, grid_size_y=1):
    # Define grid parameters
    center_x = center_point[0]  # Center x-coordinate of the grid
    center_y = center_point[1]  # Center y-coordinate of the grid
    min_x = min_point[0]  # Minimum x-coordinate
    max_x = max_point[0]  # Maximum x-coordinate
    min_y = min_point[1]  # Minimum y-coordinate
    max_y = max_point[1]  # Maximum y-coordinate

    # Calculate the number of grid lines in each direction
    num_x_lines_left = int((center_x - min_x) / grid_size_x)
    num_x_lines_right = int((max_x - center_x) / grid_size_x)
    num_y_lines_top = int((max_y - center_y) / grid_size_y)
    num_y_lines_bottom = int((center_y - min_y) / grid_size_y)

    # Generate data points for the grid and centers of squares
    grid_data = {'gridID': [], 'gridX': [], 'gridY': [], 'centerX': [], 'centerY': []}
    current_id = 0
    for i in range(-(num_x_lines_left + 1), num_x_lines_right + 1):
        for j in range(-(num_y_lines_bottom + 1), num_y_lines_top + 1):
            current_id += 1
            x = center_x + i * grid_size_x
            y = center_y + j * grid_size_y
            grid_data['gridID'].append(current_id)
            grid_data['gridX'].append(x)
            grid_data['gridY'].append(y)
            grid_data['centerX'].append(x + grid_size_x / 2)
            grid_data['centerY'].append(y + grid_size_y / 2)

    return pd.DataFrame(grid_data)


def assign_points_to_grid(points_df, grid_df, new_column_grid_df_name_and_property=[('index', 'pointIndices')]):
    # Initialize a new column in the grid DataFrame to store point indices
    for name_and_property in new_column_grid_df_name_and_property:
        grid_df[name_and_property[1]] = None

    for index, grid_row in grid_df.iterrows():
        x1, y1 = grid_row['gridX'], grid_row['gridY']
        x2, y2 = x1 + grid_row['gridSizeX'], y1 + grid_row['gridSizeY']

        # Find the points within the current grid cell
        points_in_grid = points_df[(points_df['x'] >= x1) & (points_df['x'] < x2) &
                                   (points_df['y'] >= y1) & (points_df['y'] < y2)]

        for name_and_property in new_column_grid_df_name_and_property:
            if name_and_property[0] == 'index':
                grid_df.at[index, name_and_property[1]] = points_in_grid.index
            else:
                grid_df.at[index, name_and_property[1]] = points_in_grid[name_and_property[0]]

    return grid_df


def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0


def create_file_folder_structure(output_path, folder_name):
    # path created for the data of the given input file
    file_folder_path = os.path.join(output_path, folder_name)
    # folder for output file for exports of corresponding input file
    file_folder_output_path = os.path.join(file_folder_path, "output")

    print(f"Using '{folder_name}' as input.")
    # creates a folder for the corresponding input file, where data gets saved
    if not os.path.exists(file_folder_path):
        # If not, create it
        os.makedirs(file_folder_path)
        os.makedirs(file_folder_output_path)
        print(f"Folder '{file_folder_path}' created.")
    else:
        if not os.path.exists(file_folder_output_path):
            os.makedirs(file_folder_output_path)

        print(f"Folder '{file_folder_path}' already exists.")

    return file_folder_path
