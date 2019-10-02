import pandas as pd

class Helper():
    @staticmethod
    def load_csv(filepath):
        return pd.read_csv(filepath)
