import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import time
import os 
from pathlib import Path


def plot_Reco_vs_POP(listREC, listPOP, name, epoch, num_epochs, model):
    # put in a row
    lrecommended = np.hstack(listREC)
    drec =  pd.DataFrame(lrecommended, columns=['itemrec'])
    lPopular = np.hstack(listPOP)
    dpop =  pd.DataFrame(lrecommended, columns=['itempop'])
    
    drec1 = drec.groupby(['itemrec'])['itemrec'].agg(cuenta='count').sort_values(['cuenta'], ascending=False).reset_index()
  
    fig = px.bar(drec1, x='itemrec', y='cuenta', color='cuenta', title="EPOCH:"+ epoch +" - Recommended items - number of different items:"+str(len(set(lrecommended)))+" <br> Items also in popularity list:"+str(len(set(set(lrecommended) & set(lPopular)))) + " - model:" + model)
    fig.update_xaxes(type='category')  
    
    if (int(epoch) >= (num_epochs-1)):
        if not "logs" in str(os.getcwd()):
            os.chdir(os.getcwd() / Path(f"logs"))
        fig.write_image(name)
        time.sleep(2)
    # fig.show()

