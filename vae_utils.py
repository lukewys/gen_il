from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from conv_vae_model import VAE
import train_utils

torch.manual_seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 500
NUM_WORKERS = 8
SAMPLE_NUM = 128
SAMPLE_Z = torch.randn(SAMPLE_NUM, 20).to(device)
TOTAL_EPOCH = 100

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(model_assets, train_data, train_extend):
    total_epoch = int(np.round(train_extend * TOTAL_EPOCH))
    model, optimizer = model_assets
    model.train()
    for epoch in range(total_epoch):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_data):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_data.dataset)))

    return model, optimizer


def train_with_teacher(new_model_assets, old_model_assets, steps, **kwargs):
    model, optimizer = new_model_assets
    model.train()
    train_loss = []
    for batch_idx in range(steps):
        data = gen_data(old_model_assets, BATCH_SIZE, 1).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        if batch_idx % 100 == 0:
            print('====> Step: {} Average loss: {:.4f}'.format(
                batch_idx, np.mean(train_loss) / BATCH_SIZE))

    return model, optimizer


def get_train_data(batch_size=BATCH_SIZE):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist_data/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist_data/', train=False, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ])),
        batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader


def get_model_assets(model_assets=None, reset_model=True):
    if reset_model:
        model = VAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        return model, optimizer
    else:
        return model_assets


def get_train_data_next_iter(train_data, data_generated, add_old_dataset=False, keep_portion=1.0):
    return train_utils.get_train_data_next_iter(train_data, data_generated, add_old_dataset=add_old_dataset,
                                                keep_portion=keep_portion, batch_size=BATCH_SIZE,
                                                num_workers=NUM_WORKERS)


def gen_data(model_assets, gen_batch_size, gen_num_batch, **kwargs):
    data_all = []
    model, optimizer = model_assets
    model.eval()
    with torch.no_grad():
        for _ in range(gen_num_batch):
            sample = torch.randn(gen_batch_size, 20).to(device)
            sample = model.decode(sample).cpu()
            data_all.append(sample)
    data_all = torch.cat(data_all, dim=0)
    return data_all


def save_sample(model_assets, log_dir, iteration):
    model, optimizer = model_assets
    with torch.no_grad():
        sample = model.decode(SAMPLE_Z).cpu()
    save_image(sample.view(SAMPLE_NUM, 1, 32, 32), f'{log_dir}/sample_iter_{iteration}' + '.png')
