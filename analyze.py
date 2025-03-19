from glob import glob
from scipy.io import readsav


CONVS = [20, 25, 35, 45]

data_root = "/NAS/data/halpha"

for conv in CONVS:
    data_dir = f"{data_root}/0_{conv}"
    list_data = glob(f"{data_dir}/*.sav")
    print(f"Number of data files in {data_dir}: {len(list_data)}")
