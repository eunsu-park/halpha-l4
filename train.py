import os
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

data_path = f"{experiment_dir}/data_{options.num_inp}.h5"
with h5py.File(data_path, "w") as f:
    f.create_dataset("train_inp", data=train_inp.cpu().detach().numpy())
    f.create_dataset("train_tar", data=train_tar.cpu().detach().numpy())
    f.create_dataset("test_inp", data=test_inp.cpu().detach().numpy())
    f.create_dataset("test_tar", data=test_tar.cpu().detach().numpy())

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
        network.eval()
        out = network(test_inp)
        loss = criterion(out, test_tar)
        metric = metric_function(out, test_tar)
        test_losses.append(loss.item())
        test_metrics.append(metric.item())
        print(f"Epochs {epochs}, Loss (Log-Cosh) {loss.item():.2f}, Metric (MSE) {metric.item():.2f}")
        network.train()


torch.save(network.state_dict(), f"{experiment_dir}/model.pth")

x_train = np.arange(1, len(train_losses)+1)
x_test = np.arange(100, len(train_losses)+1, 100)

plt.figure(figsize=(15, 10))
plt.plot(x_train, train_losses, color="red", label="Train Loss (Log-Cosh)")
plt.plot(x_test, test_losses, color="blue", label="Test Loss (Log-Cosh)")
plt.plot(x_test, train_metrics, color="green", label="Train Metric (MSE)")
plt.plot(x_test, test_metrics, color="orange", label="Test Metric (MSE)")
plt.title("Loss and Metric")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.savefig(f"{experiment_dir}/loss_and_metric.png", dpi=200)
plt.close()

network.eval()
out = network(test_inp)
inp = test_inp
tar = test_tar

inp = inp.cpu().detach().numpy()
tar = tar.cpu().detach().numpy()
out = out.cpu().detach().numpy()

inp = custom_data.denormalize(inp)
tar = custom_data.denormalize(tar)
out = custom_data.denormalize(out)
dif = out - tar
rel_dif = (out - tar) / tar

print(inp.shape, tar.shape, out.shape)

plot_dir = f"{experiment_dir}/plot"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

for idx in range(100) :

    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Comparison, Index {idx}")

    plt.subplot(2, 3, 1)
    plt.plot(inp[idx], color="red")
    plt.title("Input, I")

    plt.subplot(2, 3, 2)
    plt.plot(tar[idx], color="blue")
    plt.title("Target, T")

    plt.subplot(2, 3, 3)
    plt.plot(out[idx], color="green")
    plt.title("Model Output, O")

    plt.subplot(2, 3, 4)
    plt.plot(tar[idx], color="blue", label="Target")
    plt.plot(out[idx], color="green", label="Model Output")
    plt.title("T & O")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(dif[idx], color="orange")
    plt.title(f"Error, O - T")

    plt.subplot(2, 3, 6)
    plt.plot(rel_dif[idx], color="purple")
    plt.title(f"Relative Error, (O - T) / T")

    plt.savefig(f"{plot_dir}/comparison_{idx}.png", dpi=200)
    plt.close()

if options.data_mode == "index" :

    size_i = custom_data.size_i
    size_j = custom_data.size_j

    inp = reflatten_data(inp, size_i, size_j)
    tar = reflatten_data(tar, size_i, size_j)
    out = reflatten_data(out, size_i, size_j)

    save_path = f"{experiment_dir}/result.h5"

    with h5py.File(save_path, "w") as f :
        f.create_dataset("inp", data=inp)
        f.create_dataset("tar", data=tar)
        f.create_dataset("out", data=out)
        f.create_dataset("dif", data=dif)
        f.create_dataset("rel_dif", data=rel_dif)
        f.create_dataset("train_losses", data=train_losses)
        f.create_dataset("test_losses", data=test_losses)
        f.create_dataset("train_metrics", data=train_metrics)
        f.create_dataset("test_metrics", data=test_metrics)


