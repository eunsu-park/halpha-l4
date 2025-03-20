import os
from glob import glob
import h5py
import numpy as np
import torch


def flatten_data(data):
    return data.reshape(-1, data.shape[-1])

def reflatten_data(data, size_i, size_j):
    return data.reshape(size_i, size_j, data.shape[-1])


class CustomData:
    def __init__(self, options):
        data_dir = options.data_dir
        self.list_data = sorted(glob(os.path.join(data_dir, "*.h5")))
        num_inp = options.num_inp
        data_mode = options.data_mode
        if data_mode == "index" :
            test_index = options.test_index

            train_files = self.list_data.copy()
            test_file_path = train_files.pop(test_index)

            train_inp = []
            train_tar = []

            for data_path in train_files:
                with h5py.File(data_path, "r") as f:
                    inp = f[f"data_{num_inp}"][:]
                    tar = f[f"data_original"][:]

                    inp = flatten_data(inp)
                    tar = flatten_data(tar)

                    train_inp.append(inp)
                    train_tar.append(tar)

            train_inp = np.concatenate(train_inp, axis=0)
            train_tar = np.concatenate(train_tar, axis=0)
            self.median = np.median(np.concatenate([train_inp, train_tar], axis=1))
            self.std = np.std(np.concatenate([train_inp, train_tar], axis=1))

            train_inp = self.normalize(train_inp)
            train_tar = self.normalize(train_tar)

            with h5py.File(test_file_path, "r") as f:
                test_inp = f[f"data_{num_inp}"][:]
                test_tar = f[f"data_original"][:]

                self.size_i = test_tar.shape[0]
                self.size_j = test_tar.shape[1]

                test_inp = flatten_data(test_inp)
                test_tar = flatten_data(test_tar)

                test_inp = self.normalize(test_inp)
                test_tar = self.normalize(test_tar)

            self.train_inp = torch.tensor(train_inp, dtype=torch.float32)
            self.train_tar = torch.tensor(train_tar, dtype=torch.float32)
            self.test_inp = torch.tensor(test_inp, dtype=torch.float32)
            self.test_tar = torch.tensor(test_tar, dtype=torch.float32)

        elif data_mode == "random" :
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

    HOME = os.environ["HOME"]

    options.data_dir = os.path.join(HOME, "Data", "halpha", "dataset")
    # options.data_dir = os.path.join("D:\\Data", "halpha", "dataset")

    data = CustomData(options)
    train_inp, train_tar, test_inp, test_tar = data.get_data()
    print(train_inp.shape, train_tar.shape, test_inp.shape, test_tar.shape)
