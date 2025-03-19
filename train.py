import os
import matplotlib.pyplot as plt
from options import Options
options = Options().parse()

from pipeline import CustomData
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

network = CustomNetwork(options)
print(network)
network = network.to(device)

criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

def lambda_rule(epoch):
    return 1.0 - max(0, epoch + 1 - options.nb_epochs) / float(options.nb_epochs_decay + 1)    
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda_rule)

train_losses = []
test_losses = []

epochs = 0
while epochs <= options.nb_epochs + options.nb_epochs_decay :
    network.train()
    optimizer.zero_grad()
    out = network(train_inp)
    loss = criterion(out, train_tar)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    epochs += 1
    scheduler.step()

    if epochs % 100 == 0:
        network.eval()
        out = network(test_inp)
        loss = criterion(out, test_tar)
        test_losses.append(loss.item())
        print(f"Epochs {epochs}, Loss {loss.item()}")
        network.train()

save_dir = f"{options.output_dir}/{options.experiment}/out_{options.num_inp}"
os.makedirs(save_dir, exist_ok=True)
torch.save(network.state_dict(), f"{save_dir}/model.pth")

plt.figure(figsize=(15, 10))
plt.plot(train_losses, color="red", label="Train Loss")
plt.plot(np.arange(0, len(test_losses) * 100, 100), test_losses, color="blue", label="Test Loss")
plt.legend()
plt.savefig(f"{save_dir}/loss.png", dpi=200)
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


for idx in range(inp.shape[0]) :

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

    plt.savefig(f"{save_dir}/comparison_{idx}.png", dpi=200)
    plt.close()



