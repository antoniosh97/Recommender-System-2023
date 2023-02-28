import os
import sys

from datetime import datetime, timedelta
import time
import random
import time
import math

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np

import csv
import json
import gzip
from pathlib import Path

from urllib.request import urlopen
from itertools import chain
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go
import logging
from tqdm import tqdm

import scipy.sparse as sp
from typing import Tuple, Dict, Any, List
from tqdm import tqdm, trange
from IPython import embed
from IPython.display import display, HTML, clear_output

from statistics import mean
import ipywidgets as widgets
import requests

import logs
import dataset
import pointdata
import model_fm
import model_random
import model_nfc
import sampling
import results
import multiprocessing as mp

class Main():
    def __init__(self):

        self.device    = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_cores = mp.cpu_count()
        if self.num_cores > 1:
            self.num_cores = self.num_cores - 1
        self.num_cores = int(8)
        
        self.ini_time   = datetime.now()
        self.log        = logs.Logs()
        self.ds         = dataset.DataSet()
        self.spl        = sampling.Sample()
        self.res        = results.Results()

        self.hparams = {
            'batch_size':64,
            'num_epochs':12,
            'hidden_size': 32,
            'learning_rate':1e-4,
        }
        self.min_reviews, self.min_usuarios = [6,6]
        self.col_names =   {"col_id_reviewer": "reviewerID",
                            "col_id_product": "asin",
                            "col_unix_time": "unixReviewTime",
                            "col_rating": "overall",
                            "col_timestamp": "timestamp",
                            "col_year": "year"}
        self.sampling_method = "neg_sample"
        self.execution_path = os.getcwd()
        self.train_x, self.test_x = [], []
    

    def start(self):

        # Logs
        self.base_dir_name = os.getcwd()
        self.exec_path = os.getcwd()
        tb_fm, tb_rnd = self.log.def_log(self.base_dir_name, self.exec_path)
        
        # Dataset
        dataset_path = self.exec_path + "/3_DataPreparation/"
        df = self.ds.readDataSet(dataset_path, self.min_reviews, self.min_usuarios, nrows=None) 
        self.log.save_data_configuration(str(df.nunique()))
        data, dims = self.ds.getDims(df, self.col_names, True)
        
        # Prepare data for training
        self.train_x, self.test_x = self.split_train_test(data, dims[0])
        self.train_x = self.train_x[:, :2]
        dims = dims[:2]
        
        # Sampling Strategy
        print("Start: Neg Sampling")
        self.train_x, rating_mat = self.spl.ng_sample(self.train_x, dims)
        dims[-1]-dims[0]
        print("End: Neg Sampling")

        print("Start: zero_positions Sampling")
        zero_positions = self.spl.zero_positions_mode(1, rating_mat, self.log, showtime=False)
        print(f"zero_positions: {str(len(zero_positions))}") #287868348
        print("End: zero_positions Sampling")

        print("Start: items2compute Sampling")
        chunk_size2 = len(zero_positions) // self.num_cores
        print(f"Getting chunks...")
        chunks2 = [(int(round(i*chunk_size2)), int(round((i+1)*chunk_size2))) for i in range(self.num_cores)]
        print(f"NrChunks: {str(len(chunks2))}")
        
        ################################################################################################
        print(f"Starting parallel processing...")

        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        idx = 0
        for (start, end) in chunks2:
            #end = 100000
            process_name = f"RS-Process_{str(idx)}_{os.getpid()}_{int(time.time())}"
            process = mp.Process(target=self.items2compV3, args=(idx, dims, zero_positions, start, end, return_dict), name=process_name)
            processes.append(process)
            process.start()
            idx += 1
            
            #if idx > 2:
            #    break

        ################################################################################################
        # Get the return of each process

        items2compute =  []
        p_already_collected = [False for _ in processes]
        getting_return = 1
        while getting_return == 1:
            idx = 0
            check_process_alive = False
            for proces in processes:
                if p_already_collected[idx] == False:
                    if proces.is_alive():
                        check_process_alive = True 
                    else:
                        items2compute = items2compute + return_dict[idx]
                        p_already_collected[idx] = True
                idx += 1
            if check_process_alive == False:
                getting_return = 0
        print("All processes finished")
        print(f"items2compute: {str(len(items2compute))}") 
        del p_already_collected, getting_return, idx, check_process_alive, proces, processes, return_dict
        print("End: items2compute Sampling")

        ################################################################################################

        print("Start: build_test_set")
        self.test_x = self.build_test_set(items2compute, self.test_x)
        print("End: build_test_set")
        
        print("Start: Create Models: FM, Random, NFC")
        train_dataset = pointdata.PointData(self.train_x, dims)
        dims = train_dataset.dims
        model = model_fm.FactorizationMachineModel(dims, self.hparams['hidden_size']).to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.hparams['learning_rate'])

        rnd_model = model_random.RandomModel(dims)

        """
        num_users = len(dims[0])
        num_items = len(dims[1])
        nfc_model = nfc_model.NCF(num_users, num_items, self.hparams['hidden_size'], self.hparams['batch_size'])
        """

        data_loader = DataLoader(train_dataset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0)
        print("End: Create Models: FM, Random, NFC")
        
        print("Start: Epochs")
        topk = 10
        for epoch_i in range(self.hparams['num_epochs']):
            #data_loader.dataset.negative_sampling()
            train_loss = self.train_one_epoch(model, optimizer, data_loader, criterion, self.device)
            hr, ndcg, coverage = self.test(model, self.train_x, self.test_x, self.device, topk=topk)
            
            print(self.log.save_data_configuration(f'MODEL: FACTORIZATION MACHINE'))
            print(self.log.save_data_configuration(f'epoch {epoch_i}:'))
            print(self.log.save_data_configuration(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f}, Coverage@:{coverage:.2f}%'))
            print('\n')
        
            tb_fm.add_scalar('train/loss', train_loss, epoch_i)
            tb_fm.add_scalar('eval/HR@{topk}', hr, epoch_i)
            tb_fm.add_scalar('eval/NDCG@{topk}', ndcg, epoch_i)

            hr, ndcg, coverage  = self.test(rnd_model, self.train_x, self.test_x, self.device, topk=topk)

            print(self.log.save_data_configuration(f'MODEL: RANDOM'))
            print(self.log.save_data_configuration(f'epoch {epoch_i}:'))
            print(self.log.save_data_configuration(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f}, Coverage@:{coverage:.2f}%'))
            print('\n')
        
            tb_rnd.add_scalar('eval/HR@{topk}', hr, epoch_i)
            tb_rnd.add_scalar('eval/NDCG@{topk}', ndcg, epoch_i)
        
        print("End: Epochs")
        #os.chdir(os.getcwd() / Path("4_Modelling"))
        #%tensorboard --logdir run_tensorboard
       
        print("End of Method Start")

        #Calc time of execution
        self.efe(self.ini_time)

    #>>> End of Method Start

    def efe(self, startime):
        end_time = datetime.now()
        time_dif = end_time - startime
        seconds = time_dif.seconds
        print(f" ")
        secmin = ""
        if seconds > 60: 
            seconds = seconds / 60
            secmin = "minutes"
        elif seconds <= 60: 
            secmin = "seconds"
        print(f"Executed in {seconds} {secmin}")


    def items2compV3(self, idx, dims, zero_positions, start, end, return_dict):
        print(f"Processing Chunk from {str(start)} to {str(end)}")
        chunk = zero_positions[start:end]
        out = []

        mask = chunk[:, 1] >= dims[0]
        chunk = chunk[mask]
        usuarios = chunk[:, 0]
        lista_longitud_zeros = np.bincount(usuarios, minlength=len(set(usuarios)))
        list_of_lists = [list() for i in range(len(lista_longitud_zeros))]

        for i, length in enumerate(lista_longitud_zeros):
            list_of_lists[i] = [0]*length

        #out = np.array([np.array(x) for x in list_of_lists])
        out = [x for x in list_of_lists]
        total_idx = len(out)

        for user in range(dims[0]):
            if user < total_idx:
                try:
                    aux = chunk[chunk[:, 0] == user][:, 1]
                    out[user] = aux
                except IndexError:
                    None
        
        return_dict[idx] = out
        return return_dict
        

    def items2compV2(self, dims, zero_positions, start, end):
        print(f"Processing Chunk from {str(start)} to {str(end)}")
        chunk = zero_positions[start:end]
        out = []

        user_indices = np.arange(dims[0])
        user_indices = chunk[:, 0]
        item_indices = chunk[:, 1]
        for user in trange(dims[0]):
            user_mask = user_indices == user
            user_zero_item_indices = item_indices[user_mask]
            user_item_indices = user_zero_item_indices[user_zero_item_indices >= dims[0]]
            out.append(list(user_item_indices))
        return out


    def items2computes(self, dims, zero_positions, showtime=False, mode=0):
        if showtime:
            ini_time   = datetime.now()

        print(f"zero_positions: {str(len(zero_positions))}") #287868348
        print(f"dims[0]: {str(len(dims[0]))}")
        items2compute = []

        if mode == 0:
            for user in dims[0]:
                aux = zero_positions[zero_positions[:, 0] == user][:, 1]
                items2compute.append(aux[aux >= dims[0]])
            
        elif mode == 1:
            user_indices = np.arange(dims[0])
            user_indices = zero_positions[:, 0]
            item_indices = zero_positions[:, 1]
            for user in trange(dims[0]):
                user_mask = user_indices == user
                user_zero_item_indices = item_indices[user_mask]
                user_item_indices = user_zero_item_indices[user_zero_item_indices >= dims[0]]
                items2compute.append(user_item_indices)

        elif mode == 2:
            user_indices = np.arange(dims[0])
            user_indices = zero_positions[:, 0]
            item_indices = zero_positions[:, 1]
            for user in trange(dims[0]):
                user_mask = user_indices == user
                aux = item_indices[user_mask]
                #aux = zero_positions[zero_positions[:, 0] == user][:, 1]
                user_items = chain.from_iterable([aux[aux >= dims[0]]])
                items2compute.append(list(user_items))

        if showtime:
            end_time = datetime.now()
            time_dif = end_time - self.ini_time
            seconds = time_dif.seconds
            if seconds > 60: 
                seconds = seconds / 60
                print(f"items2compute - Executed in {seconds} minutos")
            elif seconds <= 60: 
                print(f"items2compute - Executed in {seconds} seconds")

        return items2compute
    
    def split_train_test(self,
                         data: np.ndarray,
                         n_users: int) -> Tuple[np.ndarray, np.ndarray]:
        # Split and remove timestamp
        train_x, test_x = [], []
        for u in trange(n_users, desc='spliting train/test and removing timestamp...'):
            user_data = data[data[:, 0] == u]
            sorted_data = user_data[user_data[:, -1].argsort()]
            if len(sorted_data) > 0:
                if len(sorted_data) == 1:
                    train_x.append(sorted_data[0][:-1])
                else:
                    train_x.append(sorted_data[:-1][:, :-1])
                    test_x.append(sorted_data[-1][:-1])
        return np.vstack(train_x), np.stack(test_x)
    
    def build_test_set(self, itemsnoninteracted:list, gt_test_interactions: np.ndarray) -> list:
        #max_users, max_items = dims # number users (943), number items (2625)
        test_set = []
        for pair, negatives in tqdm(zip(gt_test_interactions, itemsnoninteracted), desc="BUILDING TEST SET..."):
            # APPEND TEST SETS FOR SINGLE USER
            negatives = np.delete(negatives, np.where(negatives == pair[1]))
            single_user_test_set = np.vstack([pair, ] * (len(negatives)+1))
            single_user_test_set[:, 1][1:] = negatives
            test_set.append(single_user_test_set.copy())
        return test_set
    
    def train_one_epoch(self, 
                        model: torch.nn.Module,
                        optimizer: torch.optim,
                        data_loader: torch.utils.data.DataLoader,
                        criterion: torch.nn.functional,
                        device: torch.device) -> float:
        model.train()
        total_loss = []

        for i, (interactions, targets) in enumerate(data_loader):
            interactions = interactions.to(device)
            targets = targets.to(device)

            predictions = model(interactions)
        
            loss = criterion(predictions, targets.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())

        return mean(total_loss)

    def test(self, 
             model: torch.nn.Module,
             train_x: np.ndarray,
             test_x: np.ndarray,
             device: torch.device,
             topk: int=10) -> Tuple[float, float]:
        # Test the HR and NDCG for the model @topK
        model.eval()
        res = results.Results()

        HR, NDCG = [], []
        all_reco_list = np.array([])
        cov_sum_recolist_prod_by_user = 0
        for user_test in test_x:
            gt_item = user_test[0][1]
            predictions = model.predict(user_test, device)
            #_, indices = torch.topk(predictions, topk)
            _, indices = torch.topk(predictions, min(topk, predictions.size()[0]))
            recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]

            #Sum unique recommended products by user to calculate coverage
            all_reco_list = np.unique(np.hstack([all_reco_list, recommend_list]))
 
            HR.append(res.getHitRatio(recommend_list, gt_item))
            NDCG.append(res.getNDCG(recommend_list, gt_item))
        
        #Calculate Coverage Metric
        cov_sum_recolist_prod_by_user += len(np.unique(all_reco_list))
        count_all_items_train = len(np.unique(train_x[:,1])) 
        coverage = (cov_sum_recolist_prod_by_user *100 ) / count_all_items_train
            
        return mean(HR), mean(NDCG), coverage

#<<< End of Class Main

if __name__ == '__main__':
    main = Main()
    main.start()
