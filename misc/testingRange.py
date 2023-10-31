import numpy as np

if __name__ == '__main__':
    # array = np.load(r"C:\Users\koenig\OneDrive - Students RWTH Aachen University\Bachelorarbeit\GitHub\twoP\Playground\Luca\PlaygoundProject\data\output\2023-10-31_10-12-13_umap_cluster_output.npy", allow_pickle=True)
    # print(array)

    customColorPalette = [colorPalette[int(int(i * (len(colorPalette) - 1) / len(uniqueFactors)))] for i in
                          range(len(uniqueFactors))]

