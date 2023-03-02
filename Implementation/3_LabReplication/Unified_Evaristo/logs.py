import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil

class Logs():
    def __init__(self, exec_path, sm, ml=True):
        self.exec_path = exec_path
        self.log_dir = "4_Modelling"
        self.sampling_method = sm
        self.multi_logs = ml

        self.save_data_dir = "data_config_" + self.sampling_method +".txt"
        self.path_save_data_dir = f'{self.exec_path}/{self.log_dir}/{self.save_data_dir}'
        
        if self.multi_logs == False:
            if os.path.isfile(self.path_save_data_dir):
                open(self.path_save_data_dir, "w").close()
        
        text = datetime.now().strftime("%Y.%m.%d-%H:%M:%S") + " - RS-Execution Result"
        with open(self.path_save_data_dir, "a") as data_file:
            data_file.write(text+"\n\n")

    def def_log(self):
        #self.sampling_method = "neg_sample" #os.listdir()[3].split(".")[-2][3:].split("_")[-1]
        #old_path = os.getcwd()
        #os.chdir("..")
        #%load_ext tensorboard

        logs_base_dir = "runs_" + self.sampling_method
        os.environ["run_tensorboard"] = logs_base_dir
        dir_path = f'{self.exec_path}/{self.log_dir}/{logs_base_dir}'

        if self.multi_logs == False:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

        os.makedirs(dir_path, exist_ok=True)
        tb_fm = SummaryWriter(log_dir=f'{dir_path}/{logs_base_dir}_FM/')
        tb_rnd = SummaryWriter(log_dir=f'{dir_path}/{logs_base_dir}_RANDOM/')
        #tb_nfc = SummaryWriter(log_dir=f'{self.exec_path}/{self.log_dir}/{logs_base_dir}/{logs_base_dir}_NFC/')
        return tb_fm, tb_rnd

    def save_data_configuration(self, text):
        #save_data_dir = "data_config_" + self.sampling_method +".txt"
        #path = f'{self.exec_path}/{self.log_dir}/{save_data_dir}'
        with open(self.path_save_data_dir, "a") as data_file:
            data_file.write(text+"\n")
        return text

