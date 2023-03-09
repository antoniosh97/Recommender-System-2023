import pandas as pd
import numpy as np

class DataSet():
    def __init__(self):
        self.data_path = "/data/"
    
    def readDataSet(self, path, min_reviews, min_usuarios, nrows=None):
        csv_filename = str(path + self.data_path) + "interactions_minR" + str(min_reviews) + "_minU" + str(min_usuarios) + ".csv"

        if nrows == None:
            df = pd.read_csv(csv_filename)
        else:
            df = pd.read_csv(csv_filename, nrows=nrows)
        return df
    
    def getDims(self, df, cols):
        data = df[[*cols.values()][:4]].astype('int32').to_numpy()

        add_dims=0
        for i in range(data.shape[1] - 2):
            # MAKE IT START BY 0
            data[:, i] -= np.min(data[:, i])
            # RE-INDEX
            data[:, i] += add_dims
            add_dims = np.max(data[:, i]) + 1
        dims = np.max(data, axis=0) + 1
        return data, dims

