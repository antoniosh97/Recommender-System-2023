import os
import sys

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

class Logs():
    def __init__(self):
        self.sampling_method = "NEG_SAMPL"
        self.exec_path = os.getcwd()

    def def_log(self, base_dir_name, exec_path):
        #self.sampling_method = "neg_sample" #os.listdir()[3].split(".")[-2][3:].split("_")[-1]
        #old_path = os.getcwd()
        #os.chdir("..")
        
        #%load_ext tensorboard

        logs_base_dir = "runs_" + base_dir_name
        os.environ["run_tensorboard"] = logs_base_dir

        os.makedirs(f'{exec_path}/{"4_Modelling"}/{logs_base_dir}', exist_ok=True)
        tb_fm = SummaryWriter(log_dir=f'{exec_path}/{"4_Modelling"}/{logs_base_dir}/{logs_base_dir}_FM/')
        tb_rnd = SummaryWriter(log_dir=f'{exec_path}/{"4_Modelling"}/{logs_base_dir}/{logs_base_dir}_RANDOM/')
        tb_nfc = SummaryWriter(log_dir=f'{exec_path}/{"4_Modelling"}/{logs_base_dir}/{logs_base_dir}_NFC/')
        return tb_fm, tb_rnd

    def save_data_configuration(self, text):
        save_data_dir = "data_config_" + self.sampling_method +".txt"
        path = f'{self.exec_path}/{"4_Modelling"}/{save_data_dir}'
        with open(path, "a") as data_file:
            data_file.write(text+"\n")
        return text
