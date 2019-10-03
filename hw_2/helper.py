import pandas as pd
import numpy as np

class Helper():
    @staticmethod
    def load_csv(filepath):
        return pd.read_csv(filepath)

    @staticmethod
    def load_csv_to_ndarray(filepath):
        with open(filepath, 'r') as f:
            header = f.readline()

        # Read using numpy lib
        data = np.genfromtxt(filepath, dtype=float, delimiter=',')

        # Return all rows except the first one
        return header.strip('\n').split(','), data[1:len(data)]
