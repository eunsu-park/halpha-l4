import os
from glob import glob
import h5py
from scipy.io import readsav
import numpy as np
import matplotlib.pyplot as plt


def read_sav(file_path):
    sav = readsav(file_path)
    wl1 = sav["wl1"]
    img1 = sav["img1"]
    wl2 = sav["wl2"]
    img2 = sav["img2"]
    return {"wl1": wl1, "img1": img1, "wl2": wl2, "img2": img2}


def flatten_data(data):
    return data.reshape(-1, data.shape[-1])


CONV = [20, 25, 35, 45]

HOME = os.environ["HOME"]
data_root = os.path.join(HOME, "Data", "halpha", "conv")
save_root = os.path.join(HOME, "Data", "halpha", "dataset")

pattern = os.path.join(data_root, "0_20", "*.sav")
data_list = glob(pattern)
print(pattern)

for file_path in data_list:
    file_name = os.path.basename(file_path)

    save_path = os.path.join(save_root, f"{os.path.splitext(file_name)[0]}.h5")

    file_20 = os.path.join(data_root, f"0_20", file_name)
    file_25 = os.path.join(data_root, f"0_25", file_name)
    file_35 = os.path.join(data_root, f"0_35", file_name)
    file_45 = os.path.join(data_root, f"0_45", file_name)

    print(file_name, os.path.exists(file_20), os.path.exists(file_25), os.path.exists(file_35), os.path.exists(file_45))

    data_20 = read_sav(file_20)
    data_25 = read_sav(file_25)
    data_35 = read_sav(file_35)
    data_45 = read_sav(file_45)

    wave_original = data_20["wl1"]
    img_original = data_20["img1"]

    wave_20 = data_20["wl2"]
    img_20 = data_20["img2"]
    num_20 = len(wave_20)

    wave_25 = data_25["wl2"]
    img_25 = data_25["img2"]
    num_25 = len(wave_25)

    wave_35 = data_35["wl2"]
    img_35 = data_35["img2"]
    num_35 = len(wave_35)

    wave_45 = data_45["wl2"]
    img_45 = data_45["img2"]
    num_45 = len(wave_45)

    print(f"# of wavelength: {len(wave_original)}, {len(wave_20)}, {len(wave_25)}, {len(wave_35)}, {len(wave_45)}")
    print(f"Data shape: {img_original.shape}, {img_20.shape}, {img_25.shape}, {img_35.shape}, {img_45.shape}")

    with h5py.File(save_path, "w") as f:
        f.create_dataset("wave_original", data=wave_original)
        f.create_dataset("data_original", data=img_original)
        f.create_dataset(f"wave_{num_20}", data=wave_20)
        f.create_dataset(f"data_{num_20}", data=img_20)
        f.create_dataset(f"wave_{num_25}", data=wave_25)
        f.create_dataset(f"data_{num_25}", data=img_25)
        f.create_dataset(f"wave_{num_35}", data=wave_35)
        f.create_dataset(f"data_{num_35}", data=img_35)
        f.create_dataset(f"wave_{num_45}", data=wave_45)
        f.create_dataset(f"data_{num_45}", data=img_45)






# for conv in CONVS:



#     save_path = os.path.join(data_root, "merged", f"{conv}.npy")

#     pattern = os.path.join(data_root, f"0_{conv}", "*.sav")
#     print(pattern)
#     file_list = glob(pattern)
#     print(f"Conv: {conv}, Number of data files: {len(file_list)}")

#     wl1 = None
#     data_list_img1 = []
#     wl2 = None
#     data_list_img2 = []

#     for file_path in file_list:
#         data = read_sav(file_path)
#         wl1 = data["wl1"]
#         img1 = data["img1"]
#         img1 = flatten_data(img1)
#         data_list_img1.append(img1)
#         wl2 = data["wl2"]
#         img2 = data["img2"]
#         img2 = flatten_data(img2)
#         data_list_img2.append(img2)

#     data_img1 = np.concatenate(data_list_img1, axis=0)
#     data_img2 = np.concatenate(data_list_img2, axis=0)

#     print(f"# of wavelength: {len(wl1)}, {len(wl2)}")
#     print(f"Data shape: {data_img1.shape}, {data_img2.shape}")