# ![DrCELL Banner](drcell/resources/banner.png) 
# DrCELL - Dimensional reduction Cluster Exploration and Labeling Library
![DrCELL Interface](misc/media/media1.gif)
![DrCELL Interface](misc/media/media2.gif)
![DrCELL Interface](misc/media/media3.gif)
## Installation Instructions:
- Download and install conda.
- Download CELL
- create conda enviroment based on the enviroment.yml file
	- open shell or CMD
	- `cd /path/to/DrCELL`
	- `conda env create -f environment.yml --name DrCELL`
## Run CELL:
- adjust data path in main.py
- start DrCELL
	- `cd /path/to/DrCELL`
	- `conda activate DrCELL`
- open CELL in extra window
 	- `python -m drcell.scripts.startApplication "/path/to/data"`
- alternativly open CELL in browser
	- `python -m drcell.scripts.startBokehServer "/path/to/data" --port 5000`
  	- open [](http://localhost:5000)http://localhost:5000 in a browser 

## How to use:
- Import your data in the format shown in main.py
- After starting the application (the first launch might take some time), you see the GUI
### General Tab
- You can select the different datasets you added in the "Data" Selection.
- With the "Color" Selection, you can select the column of your data, you want to be highlighted with color. The selectable Options can be customized with the "data_variables" in main.py
- "OR" Filter: Filters all the selected Values, by connecting them with a logical "OR". For example all red OR blue Objects. The selectable Options can be customized with the "data_variables" in main.py, and will show up with all unique values of that data column in the selection.
- "AND" Filter: Filters all the selected Values, by connecting them with a logical "AND". For example all Objects, that are red AND blue. The selectable Options can be customized with the "data_variables" in main.py, and will show up with all unique values of that data column in the selection.
- "Export Data" Button exports the current view as .npy and .mat file in the output folder. If "Export only selection" is enabled, only the data points currently on display will get exported (so for example all filtered data points won't be included) (in development). The export file can be sorted by any sortable column.
### PCA Preprocessing Tab
- The data can be Preprocessed via PCA. The reduction of PCA "n_components" can be adjusted here.
- If you select "None" as Dimensionality Reduction, the PCA is restricted to 2-Dimensions. (in development)
### Dimensional Reduction Parameters Tab
- Here you can change the Dimensional Reduction Method, as well as their parameters (in development)
- None
  - Uses just the PCA as Dimensional Reduction Method
- UMAP
  - "n_neighbors"
  - "min_dist"
- t-SNE
- PHATE
### Cluster Parameters
- "Update Cluster" Checkbox: When unchecking the box, the current clustering will be kept and not changed, when changing the parameters (does not include change of dataset). (in development)
- Selection of other Cluster Algorithms (potential future feature)
- HDBSCAN
  - unclustered data points get assigned to "cluster" -1
  - "min_cluster_size"
  - "min_sample"
  - "cluster_selection_epsilon"
  - "Allow Single-Linkage"
  - "Approximate Minimum Spanning Tree"
  - "metric"
    - "euclidean"
    - "manhattan"
    - "correlation"
    - "jaccard"
    - "hamming"
    - "chebyshev"
    - "canberra"
    - "braycurtis"
### Cluster Selection Tab
- Lets you isolate a single cluster visually. Selected via entry of an integer or the slider (currently not functional)
### Toolbar
- located on the right side of the plot
- General Tools
  - Pan
  - Box Zoom
  - Wheel Zoom
  - Save Plot
  - Reset Plot Position
  - Help
- Hover Tools
  - Data point hover tool
    - displays information of data point, when hovered over
    - the information shown, can be customized with the "display_hover_variables" variable in main.py
    - customized plot, based on data of data point (in development)
  - Grid hover tool
    - displays information about the quadrant and the data points in it, when hovered over it.
	- customized plot, based on data of data points in quadrant (in development)

### Grid Settings
- With this option you can enable a grid, separating your data points in quadrants, that can be hovered over with the hover tool and displays you additional information

### Statistics
- Some basic stats about the current selection
