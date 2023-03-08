import os
from datetime import datetime
import math
import torch
from torch.utils.data import DataLoader
import numpy as np
import logs
import dataset
import pointdata
import model_fm
import model_random
import model_pop
import model_nfc
import sampling
import exec
import savedata

class Main():
    def __init__(self):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # > Variables ------------------------------------------------
        self.test_mode = False

        self.ini_time   = datetime.now()
        self.exec_path = os.getcwd()
        self.sampling_method = "NegSampl"
        self.strategy = "TLOO"

        self.hparams = {
            'batch_size':64,
            'num_epochs':12,
            'hidden_size': 32,
            'learning_rate':1e-4,
        }

        self.pop_reco = []
        # < Variables ------------------------------------------------

        # > Classes --------------------------------------------------
        self.log = logs.Logs(exec_path=self.exec_path, sm=self.sampling_method, ml=False)
        self.spl = sampling.Sample()
        self.exec = exec.Execution()
        self.savedata = savedata.SaveData() 
        # < Classes --------------------------------------------------
        
        # > Dataset --------------------------------------------------
        self.ds = dataset.DataSet()
        self.min_reviews, self.min_usuarios = [6,6]
        self.col_names =   {"col_id_reviewer": "reviewerID",
                            "col_id_product": "asin",
                            "col_unix_time": "unixReviewTime",
                            "col_rating": "overall",
                            "col_timestamp": "timestamp",
                            "col_year": "year"}
        self.train_x, self.test_x = [], []
        # < Dataset --------------------------------------------------


    def start(self):

        # Define Logs
        tb_fm, tb_rnd, tb_pop, tb_ncf = self.log.def_log()
        
        # > Dataset ---------------------------------------------------------------------------------
        if self.test_mode:
            NrRows = 8000
        else:
            NrRows = None
 
        df = self.ds.readDataSet(self.exec_path, self.min_reviews, self.min_usuarios, nrows=NrRows)
        self.log.save_data_configuration(str(df.nunique()))
        data, dims = self.ds.getDims(df, self.col_names)
        
        if self.test_mode == True:
            print("Dim of users: {}\nDim of items: {}\nDims of unixtime: {}".format(dims[0], dims[1], dims[2]))
            print(max(data[:,0]))
            print(max(data[:,1]))
            print(data)
            print(f"len(data[:,1]): {str(len(data[:,1]))}")
        # < Dataset ---------------------------------------------------------------------------------
        
        # > Split data Training and Test-------------------------------------------------------------
        self.train_x, self.test_x = self.exec.split_train_test(data, dims[0], self.strategy)
        self.train_x = self.train_x[:, :2] #???
        dims = dims[:2]

        if self.test_mode:
            print("Train shape: ")
            print(str(self.train_x.shape))
            print("Test shape: ")
            print(str(self.test_x.shape))
        
        #self.process_res.update({"train_x.shape": self.train_x.shape})
        #self.process_res.update({"test_x.shape": self.test_x.shape})
        # < Split data Training and Test-------------------------------------------------------------
        
        # > Sampling Strategy -----------------------------------------------------------------------
        self.pop_reco = self.exec.get_pop_recons(self.train_x, dims)
        
        self.train_x, rating_mat = self.spl.ng_sample(self.train_x, dims)
        dims[-1]-dims[0] 
        if self.test_mode:
            print("Dimensions matrix:\n",dims)
            print("\nRating matrix:")
            print(rating_mat)
            print(np.count_nonzero(rating_mat.toarray())/(dims[-1]*dims[-1]))
            print(1 - np.count_nonzero(rating_mat.toarray())/(dims[-1]*dims[-1]))

            #???
            print(rating_mat.shape)
            bits = math.ceil(math.log(rating_mat.shape[0],2))
            print("rating_mat contains log2(rating_mat.shape[0]) = {} bits".format(bits))
        
        #train_x = train_x[:,[0,1,-1]] #???
        #if self.test_mode:
        #    print(train_x[:10])

        train_dataset = pointdata.PointData(self.train_x, dims)
        if self.test_mode:
            print(train_dataset[0])

        data_loader = DataLoader(train_dataset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0)

        #ng_test(rating_mat)
        print(f"Start: zero_positions")
        zero_positions = self.spl.zero_positions(rating_mat, showtime=False)
        self.log.save_data_configuration("\n"+"#"*4+"  zero_positions: all data separated by rows  "+"#"*4)
        if self.test_mode:
            print(zero_positions.shape)
            print(f"zero_positions: {str(len(zero_positions))}") 
        print(f"End: zero_positions")
        #self.process_res.update({"zero_positions size": str(len(zero_positions))})

        print("Start: items2compute")
        items2compute = self.exec.items_to_compute(zero_positions, dims)
        if self.test_mode:
            print(f"items2compute: {str(len(items2compute))}") 
        print("End: items2compute Sampling")
        # < Sampling Strategy -----------------------------------------------------------------------

        # > Build Test Set --------------------------------------------------------------------------
        #self.test_x = self.build_test_set(items2compute, self.test_x) #???
        self.test_x = self.exec.build_test_set(items2compute, self.test_x[:,:2])
        if self.test_mode:
            print(self.test_x[0])
        # > Build Test Set --------------------------------------------------------------------------
        
        # > Save Data -------------------------------------------------------------------------------
        #???
        #self.savedata.save_train(train_dataset)
        #self.savedata.save_train(self.test_x)
        #self.savedata.save_pop(self.pop_reco)
        # < Save Data -------------------------------------------------------------------------------
        
        # > Create Models --------------------------------------------------------------------------
        dims = train_dataset.dims
        fm_model = model_fm.FactorizationMachineModel(dims, self.hparams['hidden_size']).to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = torch.optim.Adam(params=fm_model.parameters(), lr=self.hparams['learning_rate'])
        rnd_model = model_random.RandomModel(dims)
        pop_model = model_pop.PopularityBasedModel(self.pop_reco)
        ncf_model = model_nfc.NCF(dims, self.hparams['hidden_size']).to(self.device)
        # < Create Models --------------------------------------------------------------------------
        
        # > Training and Test ----------------------------------------------------------------------
        # Hit Ratio: Measures whether the test item is in the top@K positions of the recommendation list
        # NDCG (Normalized Discounted Cumulative Gain). Measures the ranking quality which gives information about where in the raking is
        # Coverage: Coverage is the percent of items in the training data the model is able to recommend on a test set.

        #self.log.wr_process_res(self.process_res)
        #self.process_res.clear()

        print("Start: Epochs")
        total_items = dims[1]-dims[0] #calc coverage
        training_time_start = datetime.now()
        self.log.save_data_configuration(datetime.now().strftime("%d-%b-%Y  %H:%M"))
        topk = 10
        #fm  = np.zeros([self.hparams['num_epochs'],3])
        #rnd = np.zeros([self.hparams['num_epochs'],3])
        #pop = np.zeros([self.hparams['num_epochs'],3])
        #ncf = np.zeros([self.hparams['num_epochs'],3])

        for epoch_i in range(self.hparams['num_epochs']):
            train_loss = self.exec.train_one_epoch(fm_model, optimizer, data_loader, criterion, self.device)

            print(self.log.save_data_configuration(f'EPOCH {epoch_i}:'))
            hr, ndcg, reco_list_fm, cov_fm = 0.0, 0.0, [], 0.0
            hr, ndcg, reco_list_fm, cov_fm = self.exec.test(fm_model, self.test_x, total_items, self.device, topk=topk)
            print(self.log.save_data_configuration(f'MODEL: FM - FACTORIZATION MACHINE'))
            print(self.log.save_data_configuration(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} '))
            #fm[epoch_i] = [hr, ndcg, cov_fm]
            tb_fm.add_scalar('train/loss', train_loss, epoch_i)
            tb_fm.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_fm.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)

            hr, ndcg, reco_list_fm, cov_fm = 0.0, 0.0, [], 0.0
            hr, ndcg, reco_list_rnd, cov_rnd = self.exec.test(rnd_model, self.test_x, total_items, self.device, topk=topk)
            print(self.log.save_data_configuration(f'MODEL: RANDOM'))
            #print(self.save_data_configuration(f'epoch {epoch_i}:'))
            print(self.log.save_data_configuration(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} '))
            #rnd[epoch_i] = [hr, ndcg, cov_rnd]
            tb_rnd.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_rnd.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)

            hr, ndcg, reco_list_fm, cov_fm = 0.0, 0.0, [], 0.0
            hr, ndcg, reco_list_pop, cov_pop = self.exec.test_pop(pop_model, self.test_x, total_items, self.device, topk=topk)
            print(self.log.save_data_configuration(f'MODEL: POPULARITY-BASED'))
            #print(self.save_data_configuration(f'epoch {epoch_i}:'))
            print(self.log.save_data_configuration(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} '))
            #pop[epoch_i] = [hr, ndcg, cov_pop]
            tb_pop.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_pop.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)

            hr, ndcg, reco_list_fm, cov_fm = 0.0, 0.0, [], 0.0
            hr, ndcg, reco_list_ncf, cov_ncf = self.exec.test(ncf_model, self.test_x, total_items, self.device, topk=topk)
            print(self.log.save_data_configuration(f'MODEL: NCF - NEURAL COLLABORATIVE FILTERING'))
            #print(self.save_data_configuration(f'epoch {epoch_i}:'))
            print(self.log.save_data_configuration(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} '))
            #ncf[epoch_i] = [hr, ndcg, cov_ncf]
            tb_ncf.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_ncf.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)

            self.log.save_data_configuration("_"*65)
        
        training_time_end = datetime.now()-training_time_start
        seconds = training_time_end.seconds
        if seconds > 60: 
            seconds = seconds / 60
            secmin = "minutes"
        else:
            secmin = "seconds"
        print(self.log.save_data_configuration(f'Training duration: {seconds} {secmin}'))

        print(f"\nCoverage:")
        print(f'FM: {cov_fm:.4f}')
        print(f'RAND: {cov_rnd:.4f}')
        print(f'POP: {cov_pop:.4f}')
        print(f'NCF: {cov_ncf:.4f}')
        # < Training and Test ----------------------------------------------------------------------
      
        #Calc Total Time of execution
        txt = self.exec.efe(self.ini_time)
        print(self.log.save_data_configuration(txt))

    # > End of Method Start-------------------------------------------------------------------------

# < End of Class Main

if __name__ == '__main__':
    main = Main()
    main.start()
