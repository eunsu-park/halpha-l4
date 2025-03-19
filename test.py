from options import Options

options = Options().parse()

from pipeline import CustomData
from network import CustomNetwork

import torch

dataset = CustomData(options)
network = CustomNetwork(options)

network_path = ""
network.load_state_dict(torch.load(network_path, map_location=torch.device('cpu')))

network.eval()