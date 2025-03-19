from glob import glob
from scipy.io import readsav


def read_sav(file_path):
    sav = readsav(file_path)
    wl1 = sav["wl1"]
    img1 = sav["img1"]
    wl2 = sav["wl2"]
    img2 = sav["img2"]
    return {"wl1": wl1, "img1": img1, "wl2": wl2, "img2": img2}

CONVS = [20, 25, 35, 45]

data_root = "/NAS/data/halpha"

for conv in CONVS:
    data_dir = f"{data_root}/0_{conv}"
    list_data = glob(f"{data_dir}/*.sav")
    print(f"Number of data files in {data_dir}: {len(list_data)}")

    for data_file in list_data:
        data = read_sav(data_file)
        print(f"Data file: {data_file}")
        print(f"wl1: {data['wl1'].shape}, img1: {data['img1'].shape}")
        print(f"wl2: {data['wl2'].shape}, img2: {data['img2'].shape}")
        print()
