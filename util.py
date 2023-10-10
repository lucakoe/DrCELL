import os
import pickle

import scipy.io
import umap

umapOutDumpData = {}


def print_mat_file(file_path):
    try:
        # Load the MATLAB file
        data = scipy.io.loadmat(file_path)

        # Print the contents of the MATLAB file
        print("MATLAB File Contents:")
        for variable_name in data:
            print(f"Variable: {variable_name}")
            print(data[variable_name])
            print("\n")
    except Exception as e:
        print(f"Error: {e}")


def getUMAPOut(data, dumpPath, n_neighbors=20,
               min_dist=0.0,
               n_components=2,
               random_state=42, ):
    paramKey = (n_neighbors, min_dist, n_components, random_state)
    umapOutDfDump = {}

    # Checks the global variable if the dumpfile with this path was already called. If so instead of loading it for every call of the function it takes the data from there
    if not (dumpPath in umapOutDumpData):
        # Check if the file exists
        if os.path.exists(dumpPath):
            with open(dumpPath, 'rb') as file:
                umapOutDumpData[dumpPath] = pickle.load(file)
        else:
            # If the file doesn't exist, create it and write something to it
            with open(dumpPath, 'wb') as file:
                pickle.dump(umapOutDfDump, file)
                umapOutDumpData[dumpPath] = umapOutDfDump

            print(f"The file '{dumpPath}' has been created.")

    umapOutDfDump = umapOutDumpData[dumpPath]

    if paramKey in umapOutDfDump:
        return umapOutDfDump[paramKey]
    else:
        print(
            f"Current UMAP: n_neighbors = {n_neighbors}, min_dist = {min_dist}, n_components = {n_components}")
        umapObject = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
        )
        umapOutDfDump[paramKey] = umapObject.fit_transform(data)

        with open(dumpPath, 'wb') as file:
            umapOutDumpData[dumpPath] = umapOutDfDump
            pickle.dump(umapOutDfDump, file)
