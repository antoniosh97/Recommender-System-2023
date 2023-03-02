import os
from torch.utils.tensorboard import SummaryWriter

class Logs():
    def __init__(self, exec_path, sm):
        self.exec_path = exec_path
        self.log_dir = "4_Modelling"
        self.sampling_method = sm

    def def_log(self):
        #self.sampling_method = "neg_sample" #os.listdir()[3].split(".")[-2][3:].split("_")[-1]
        #old_path = os.getcwd()
        #os.chdir("..")
        #%load_ext tensorboard
        
        logs_base_dir = "runs_" + self.sampling_method
        os.environ["run_tensorboard"] = logs_base_dir

        os.makedirs(f'{self.exec_path}/{self.log_dir}/{logs_base_dir}', exist_ok=True)
        tb_fm = SummaryWriter(log_dir=f'{self.exec_path}/{self.log_dir}/{logs_base_dir}/{logs_base_dir}_FM/')
        tb_rnd = SummaryWriter(log_dir=f'{self.exec_path}/{self.log_dir}/{logs_base_dir}/{logs_base_dir}_RANDOM/')
        #tb_nfc = SummaryWriter(log_dir=f'{self.exec_path}/{self.log_dir}/{logs_base_dir}/{logs_base_dir}_NFC/')
        return tb_fm, tb_rnd

    def save_data_configuration(self, text):
        save_data_dir = "data_config_" + self.sampling_method +".txt"
        path = f'{self.exec_path}/{self.log_dir}/{save_data_dir}'
        with open(path, "a") as data_file:
            data_file.write(text+"\n")
        return text

