import pandas as pd
import numpy as np
import os

class DataSet():
    def __init__(self):
        self.data_path = "/data/"
    
    def readDataSet(self, path, min_reviews, min_usuarios, dataset, nrows=None):

        if dataset != "movie lens":
            csv_filename = str(path + self.data_path) + "interactions_minR" + str(min_reviews) + "_minU" + str(min_usuarios) + ".csv"
        else:
            csv_filename = str(path + self.data_path) + "interactions_movie_lens.csv"

       

        if nrows == None:
            df = pd.read_csv(csv_filename)
        else:
            df = pd.read_csv(csv_filename, nrows=nrows)
        return df
    


    def getDims(self, df, cols, dataset, col_names):

        if dataset == "movie lens":
            data=df[[col_names["col_id_reviewer"], col_names["col_id_product"], col_names["col_timestamp"]]].astype('int32').to_numpy()
       
        else: 
            data = df[[*cols.values()][:4]].astype('int32').to_numpy()

        add_dims=0

        if dataset != "movie lens":
            for i in range(data.shape[1] - 2):
                # MAKE IT START BY 0
                data[:, i] -= np.min(data[:, i])
                # RE-INDEX
                data[:, i] += add_dims
                add_dims = np.max(data[:, i]) + 1
        else:
            for i in range(data.shape[1] - 1):
                # MAKE IT START BY 0
                data[:, i] -= np.min(data[:, i])
                # RE-INDEX
                data[:, i] += add_dims
                add_dims = np.max(data[:, i]) + 1


        #dims = [np.size(np.unique(data[:, i])) for i in range(data.shape[1])]

        dims = np.max(data, axis=0) + 1
        return data, dims