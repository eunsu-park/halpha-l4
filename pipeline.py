import os
from glob import glob
import h5py
import numpy as np
import torch


class CustomData:
    def __init__(self, options):
        data_dir = options.data_dir
        self.list_data = glob(os.path.join(data_dir, "*.h5"))
        num_inp = options.num_inp
        seed = options.seed

        inps = []
        tars = []

        for data_path in self.list_data:
            with h5py.File(data_path, "r") as f:
                inp = f[f"data_{num_inp}"][:]
                tar = f[f"data_original"][:]
                inps.append(inp)
                tars.append(tar)

        inps = np.concatenate(inps, axis=0)
        tars = np.concatenate(tars, axis=0)
        self.median = np.median(np.concatenate([inps, tars], axis=1))
        self.std = np.std(np.concatenate([inps, tars], axis=1))

        inps = self.normalize(inps)
        tars = self.normalize(tars)

        # using sklearn
        # from sklearn.model_selection import train_test_split
        # train_inp, test_inp, train_tar, test_tar = train_test_split(inps, tars, test_size=0.1, random_state=seed)

        # using numpy
        indices = np.arange(inps.shape[0])
        np.random.shuffle(indices)
        perc = int(0.9 * inps.shape[0])
        train_indices = indices[:perc]
        test_indices = indices[perc:]

        train_inp = inps[train_indices]
        train_tar = tars[train_indices]
        test_inp = inps[test_indices]
        test_tar = tars[test_indices]

        self.train_inp = torch.tensor(train_inp, dtype=torch.float32)
        self.train_tar = torch.tensor(train_tar, dtype=torch.float32)
        self.test_inp = torch.tensor(test_inp, dtype=torch.float32)
        self.test_tar = torch.tensor(test_tar, dtype=torch.float32)

    def normalize(self, data):
        return (data - self.median) / self.std
    
    def denormalize(self, data):
        return data * self.std + self.median

    def get_data(self):
        return self.train_inp, self.train_tar, self.test_inp, self.test_tar



if __name__ == "__main__" :

    from options import Options
    options = Options().parse()

    options.data_dir = os.path.join("D:\\Data", "halpha", "dataset")

    data = CustomData(options)
    train_inp, train_tar, test_inp, test_tar = data.get_data()
    print(train_inp.shape, train_tar.shape, test_inp.shape, test_tar.shape)
