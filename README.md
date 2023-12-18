# Cluster Exploration and Labeling Library

## Installation Instructions:
- Download and install conda.
- Download CELL
- create conda enviroment based on the enviroment.yml file
	- open shell or CMD
	- `cd /path/to/CELL`
	- `conda env create -f environment.yml --name CELLenv`
## Run CELL
- adjust data path in main.py
- start CELL
	- `cd /path/to/CELL`
	- `conda activate CELLenv`
	- `bokeh serve .\main.py --port 5000` or `python startBokehServer.py`
  - open [](http://localhost:5000)http://localhost:5000 in a browser 
