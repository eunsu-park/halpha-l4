import torch
import torch.nn as nn

class CustomNetwork(nn.Module):
    def __init__(self, options):
        super(CustomNetwork, self).__init__()
        self.inp_size = options.num_inp
        self.tar_size = options.num_tar
        self.build()

    def build(self):
        block = []
        block += [nn.Linear(self.inp_size, 1024), nn.ReLU()]
        block += [nn.Linear(1024, 512), nn.ReLU()]
        block += [nn.Linear(512, self.tar_size)]
        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        return self.block(x)
