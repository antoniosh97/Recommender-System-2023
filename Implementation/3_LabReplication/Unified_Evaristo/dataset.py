import pandas as pd
import numpy as np

class DataSet():
    def __init__(self):
        var1 = 0
    
    def readDataSet(self, path, min_reviews, min_usuarios, nrows=None):
        csv_filename = str(path) + "interactions_minR" + str(min_reviews) + "_minU" + str(min_usuarios) + ".csv"
        print(str(csv_filename))

        if nrows != None:
            df = pd.read_csv(csv_filename, nrows=nrows)
        else:
            df = pd.read_csv(csv_filename)

        return df
    
    def getDims(self, df, cols, msg=False):
        data = df[[*cols.values()][:3]].astype('int32').to_numpy()

        add_dims=0
        for i in range(data.shape[1] - 1):  # do not affect to timestamp
            # MAKE IT START BY 0
            data[:, i] -= np.min(data[:, i])
            # RE-INDEX
            data[:, i] += add_dims
            add_dims = np.max(data[:, i]) + 1
        
        dims = np.max(data, axis=0) + 1
        if msg == True:
            print("Dim of users: {}\nDim of items: {}\nDims of unixtime: {}".format(dims[0], dims[1], dims[2]))

        return data, dims

