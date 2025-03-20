import os
import time
import h5py
import matplotlib.pyplot as plt
from options import Options
options = Options().parse()

from pipeline import CustomData, reflatten_data
from network import CustomNetwork

import torch
import random
import numpy as np

seed = options.seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device(options.device)
print(device)

custom_data = CustomData(options)
train_inp, train_tar, test_inp, test_tar = custom_data.get_data()
print(train_inp.shape, train_tar.shape, test_inp.shape, test_tar.shape)
train_inp = train_inp.to(device)
train_tar = train_tar.to(device)
test_inp = test_inp.to(device)
test_tar = test_tar.to(device)

experiment_dir = f"{options.output_dir}/{options.experiment_name}"
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

network = CustomNetwork(options)
print(network)
network = network.to(device)


def log_cosh_loss(y_pred, y):
    return torch.mean(torch.log(torch.cosh(y_pred - y)))
criterion = log_cosh_loss
metric_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=2e-4)
def lambda_rule(epoch):
    half = options.num_epochs//2
    return 1.0 - max(0, epoch + 1 - half) / float(options.num_epochs - half + 1)    
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda_rule)

train_losses = []
test_losses = []
train_metrics = []
test_metrics = []

epochs = 0
t0 = time.time()
while epochs <= options.num_epochs :
    network.train()
    optimizer.zero_grad()
    out = network(train_inp)
    loss = criterion(out, train_tar)
    metric = metric_function(out, train_tar)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    train_metrics.append(metric.item())

    epochs += 1
    scheduler.step()

    if epochs % 100 == 0:
        elapsed = time.time() - t0
        network.eval()
        test_out = network(test_inp)
        loss = criterion(test_out, test_tar)
        metric = metric_function(test_out, test_tar)
        test_losses.append(loss.item())
        test_metrics.append(metric.item())
        message = ""
        message += f"Epochs {epochs}, "
        message += f"Loss (Log-Cosh) {loss.item():.2f}, "
        message += f"Metric (MSE) {metric.item():.2f}, "
        message += f"Elapsed (Sec) {elapsed:.2f}"
        print(message)
        t0 = time.time()


torch.save(network.state_dict(), f"{experiment_dir}/model.pth")

x_train = np.arange(1, len(train_losses)+1)
x_test = np.arange(100, len(train_losses)+1, 100)

plt.figure(figsize=(15, 10))
plt.plot(x_train, train_losses, color="red", label="Train Loss (Log-Cosh)")
plt.plot(x_test, test_losses, color="blue", label="Test Loss (Log-Cosh)")
plt.plot(x_train, train_metrics, color="green", label="Train Metric (MSE)")
plt.plot(x_test, test_metrics, color="orange", label="Test Metric (MSE)")
plt.yscale("log")
plt.title("Loss and Metric")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.savefig(f"{experiment_dir}/loss_and_metric.png", dpi=200)
plt.close()

network.eval()

inp = test_inp
tar = test_tar
out = network(test_inp)
wave_inp = custom_data.wave_inp
wave_tar = custom_data.wave_tar

train_inp = train_inp.cpu().detach().numpy()
train_tar = train_tar.cpu().detach().numpy()
train_inp = custom_data.denormalize(train_inp)
train_tar = custom_data.denormalize(train_tar)

test_inp = test_inp.cpu().detach().numpy()
test_tar = test_tar.cpu().detach().numpy()
test_out = test_out.cpu().detach().numpy()
test_inp = custom_data.denormalize(test_inp)
test_tar = custom_data.denormalize(test_tar)
test_out = custom_data.denormalize(test_out)

if options.data_mode == "index" :
    test_inp = reflatten_data(test_inp, custom_data.size_i, custom_data.size_j)
    test_tar = reflatten_data(test_tar, custom_data.size_i, custom_data.size_j)
    test_out = reflatten_data(test_out, custom_data.size_i, custom_data.size_j)
test_error = test_out - test_tar
test_rel_error = test_error / test_tar

print(test_inp.shape, test_tar.shape, test_out.shape,
      test_error.shape, test_rel_error.shape)

plot_dir = f"{experiment_dir}/plot"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

for idx in range(100) :

    if options.data_mode == "index" :
        arr_i = np.random.randint(0, custom_data.size_i)
        arr_j = np.random.randint(0, custom_data.size_j)
    elif options.data_mode == "random" :
        arr_i = np.random.randint(0, inp.shape[0])

    plt.figure(figsize=(15, 10))
    if options.data_mode == "index" :
        plt.suptitle(f"Comparison, Index ({arr_i}, {arr_j})")
    elif options.data_mode == "random" :
        plt.suptitle(f"Comparison, Random ({arr_i})")

    plt.subplot(2, 3, 1)
    if options.data_mode == "index" :
        plt.plot(wave_inp, test_inp[arr_i, arr_j], color="red")
    elif options.data_mode == "random" :
        plt.plot(wave_inp, test_inp[arr_i], color="red")
    plt.title("Input, I")

    plt.subplot(2, 3, 2)
    if options.data_mode == "index" :
        plt.plot(wave_tar, test_tar[arr_i, arr_j], color="blue")
    elif options.data_mode == "random" :
        plt.plot(wave_tar, test_tar[arr_i], color="blue")
    plt.title("Target, T")

    plt.subplot(2, 3, 3)
    if options.data_mode == "index" :
        plt.plot(wave_tar, test_out[arr_i, arr_j], color="green")
    elif options.data_mode == "random" :
        plt.plot(wave_tar, test_out[arr_i], color="green")
    plt.title("Model Output, O")

    plt.subplot(2, 3, 4)
    if options.data_mode == "index" :
        plt.plot(wave_tar, test_tar[arr_i, arr_j], color="blue", label="Target")
        plt.plot(wave_tar, test_out[arr_i, arr_j], color="green", label="Model Output")
    elif options.data_mode == "random" :
        plt.plot(wave_tar, test_tar[arr_i], color="blue", label="Target")
        plt.plot(wave_tar, test_out[arr_i], color="green", label="Model Output")
    plt.title("T & O")
    plt.legend()

    plt.subplot(2, 3, 5)
    if options.data_mode == "index" :
        plt.plot(wave_tar, test_error[arr_i, arr_j], color="red")
    elif options.data_mode == "random" :
        plt.plot(wave_tar, test_error[arr_i], color="red")
    plt.title(f"Error, O - T")

    plt.subplot(2, 3, 6)
    if options.data_mode == "index" :
        plt.plot(wave_tar, test_rel_error[arr_i, arr_j], color="blue")
    elif options.data_mode == "random" :
        plt.plot(wave_tar, test_rel_error[arr_i], color="blue")
    plt.title(f"Relative Error, (O - T) / T")

    plt.savefig(f"{plot_dir}/comparison_{idx}.png", dpi=200)
    plt.close()


save_path = f"{experiment_dir}/result.h5"
with h5py.File(save_path, "w") as f:
    f.create_dataset("train_inp", data=train_inp)
    f.create_dataset("train_tar", data=train_tar)
    f.create_dataset("test_inp", data=test_inp)
    f.create_dataset("test_tar", data=test_tar)
    f.create_dataset("test_out", data=test_out)
    f.create_dataset("test_error", data=test_error)
    f.create_dataset("test_rel_error", data=test_rel_error)
    f.create_dataset("wave_inp", data=wave_inp)
    f.create_dataset("wave_tar", data=wave_tar)
    f.create_dataset("train_losses", data=train_losses)
    f.create_dataset("test_losses", data=test_losses)
    f.create_dataset("train_metrics", data=train_metrics)
    f.create_dataset("test_metrics", data=test_metrics)
