import numpy as np
from scipy.io import readsav
import torch

class CustomData:
    def __init__(self, options):
        data_type = options.data_type
        data_dir = options.data_dir
        possible_data_type = ["plage", "fil_b", "fil_r", "flare", "cr_r"]

        if data_type in possible_data_type:
            self.inp_file_path = f"{data_dir}/convol_data_{data_type}.sav"
            self.tar_file_path = f"{data_dir}/data_{data_type}.sav"
            inp = readsav(self.inp_file_path)["data"][:,:,1]
            tar = readsav(self.tar_file_path)["data"][:,:,1]
            total = np.concatenate([inp, tar], axis=1)
            self.median = np.median(total)
            self.std = np.std(total)

            perc = int(0.8 * inp.shape[0])
            train_inp = inp[:perc]
            train_tar = tar[:perc]
            test_inp = inp[perc:]
            test_tar = tar[perc:]

        elif data_type == "all" :
            train_inp = []
            train_tar = []
            test_inp = []
            test_tar = []
            total = []
            for dt in possible_data_type:
                inp = readsav(f"{data_dir}/convol_data_{dt}.sav")["data"][:,:,1]
                tar = readsav(f"{data_dir}/data_{dt}.sav")["data"][:,:,1]
                total.append(np.concatenate([inp, tar], axis=1))

                perc = int(0.8 * inp.shape[0])                
                train_inp_dt = inp[:perc]
                train_tar_dt = tar[:perc]
                test_inp_dt = inp[perc:]
                test_tar_dt = tar[perc:]

                train_inp.append(train_inp_dt)
                train_tar.append(train_tar_dt)
                test_inp.append(test_inp_dt)
                test_tar.append(test_tar_dt)

            total = np.concatenate(total, axis=1)
            self.median = np.median(total)
            self.std = np.std(total)

            train_inp = np.concatenate(train_inp, axis=0)
            train_tar = np.concatenate(train_tar, axis=0)
            test_inp = np.concatenate(test_inp, axis=0)
            test_tar = np.concatenate(test_tar, axis=0)

        else :
            raise ValueError(f"Data type {data_type} not supported")

        self.train_inp = torch.tensor(self.normalize(train_inp), dtype=torch.float32)
        self.train_tar = torch.tensor(self.normalize(train_tar), dtype=torch.float32)
        self.test_inp = torch.tensor(self.normalize(test_inp), dtype=torch.float32)
        self.test_tar = torch.tensor(self.normalize(test_tar), dtype=torch.float32)

    def normalize(self, data):
        return (data - self.median) / self.std
    
    def denormalize(self, data):
        return data * self.std + self.median

    def get_data(self):
        return self.train_inp, self.train_tar, self.test_inp, self.test_tar
