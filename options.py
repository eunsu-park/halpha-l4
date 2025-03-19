import argparse

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=1220, help="random seed")
        parser.add_argument("--experiment", type=str, default="default", help="experiment name")        
        parser.add_argument("--data_dir", type=str, default="data", help="data directory")
        parser.add_argument("--output_dir", type=str, default="output", help="output directory")
        parser.add_argument("--device", type=str, default="mps", help="device")
        parser.add_argument("--data_type", type=str, default="plage", help="data type")
        parser.add_argument("--num_inp", type=int, default=11, help="number of input channels")
        parser.add_argument("--num_tar", type=int, default=420, help="number of target channels")
        parser.add_argument("--nb_epochs", type=int, default=5000, help="number of epochs")
        parser.add_argument("--nb_epochs_decay", type=int, default=5000, help="number of epochs")

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
