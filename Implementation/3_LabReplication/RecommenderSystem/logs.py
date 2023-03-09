import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil

class Logs():
    def __init__(self, exec_path, ml=True):
        self.exec_path = exec_path
        self.multi_logs = ml
        self.log_dir = "logs"
        self.exectime = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.path_log_folder = f'{self.exec_path}/{self.log_dir}'
        self.save_data_dir = self.exectime + "_data_config.txt"
        self.path_save_data_dir = f'{self.exec_path}/{self.log_dir}/{self.save_data_dir}'

        if self.multi_logs == False:
            for filename in os.listdir(self.path_log_folder):
                if filename.startswith(self.save_data_dir[:4]):
                    os.remove(os.path.join(self.path_log_folder, filename))
            if os.path.isfile(self.path_save_data_dir):
                open(self.path_save_data_dir, "w").close()
        
        text = self.exectime + " - RS-Execution Result"
        with open(self.path_save_data_dir, "a") as data_file:
            data_file.write(text+"\n\n")

    def def_log(self):
        logs_base_dir = "runs"
        dir_path = f'{self.exec_path}/{self.log_dir}/{logs_base_dir}'

        if self.multi_logs == False:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

        os.makedirs(dir_path, exist_ok=True)
        tb_fm  = SummaryWriter(log_dir=f'{dir_path}/{logs_base_dir}_FM/')
        tb_rnd = SummaryWriter(log_dir=f'{dir_path}/{logs_base_dir}_RANDOM/')
        tb_pop = SummaryWriter(log_dir=f'{dir_path}/{logs_base_dir}_POP/')
        tb_ncf = SummaryWriter(log_dir=f'{dir_path}/{logs_base_dir}_NCF/')
        return tb_fm, tb_rnd, tb_pop, tb_ncf

    def save_data_configuration(self, text):
        with open(self.path_save_data_dir, "a") as data_file:
            data_file.write(text+"\n")
        return text
    





    def wr_process_res(self, process_res):
        col = ""
        row = ""
        for k, v in process_res.items():
            col += str(k) + ";"
            row += str(v) + ";"
        with open(self.process_res_dir, "a") as data_file:
            data_file.write(col+"\n")
            data_file.write(row+"\n")
        
    def wr_model_res(self, model_res):
        col = ""
        row = ""
        for val in model_res:
            for k, v in val.items():
                col += str(k) + ";"   
                 
            for k, v in model_res.items():
                row += str(v) + ";"
        with open(self.model_res_dir, "a") as data_file:
            data_file.write(col+"\n")
            data_file.write(row+"\n")