from __future__ import print_function
import argparse
import copy
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from conv_vae_model import VAE
import train_utils
from linear_prob_utils import LinearProbeModel

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


def mnist_subset(dataset, classes_to_use):
    # classes_to_use: a list of class
    data = dataset.data.numpy()
    targets = dataset.targets.numpy()

    data_to_keep = []
    targets_to_keep = []
    for i in range(len(targets)):
        if targets[i] in classes_to_use:
            data_to_keep.append(data[i])
            targets_to_keep.append(targets[i])

    dataset.data = torch.tensor(np.stack(data_to_keep, axis=0))
    dataset.targets = torch.tensor(np.array(targets_to_keep))
    return dataset

def get_transform(dataset_name):
    if dataset_name in ['mnist', 'fashion-mnist', 'kuzushiji']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
        ])
    elif dataset_name == 'omniglot':
        return transforms.Compose([
            transforms.ToTensor(),
            train_utils.flip_image_value,
            transforms.Resize(32),
        ])
    elif dataset_name in ['cifar10', 'cifar100', 'wikiart']:
        raise NotImplementedError  # TODO


# def get_init_data(batch_size=BATCH_SIZE, holdout_digits=None):
#     if holdout_digits is not None:
#         dataset = datasets.MNIST('./mnist_data/', train=True, download=True,
#                                  transform=transforms.Compose([
#                                      transforms.Resize(32),
#                                      transforms.ToTensor()
#                                  ])),
#         dataset = mnist_subset(dataset, [d for d in range(10) if d not in holdout_digits])
#         train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
#                                                    num_workers=NUM_WORKERS, pin_memory=True)
#     else:
#         train_loader = torch.utils.data.DataLoader(
#             datasets.MNIST('./mnist_data/', train=True, download=True,
#                            transform=transforms.Compose([
#                                transforms.Resize(32),
#                                transforms.ToTensor()
#                            ])),
#             batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('./mnist_data/', train=False, download=True,
#                        transform=transforms.Compose([
#                            transforms.Resize(32),
#                            transforms.ToTensor()
#                        ])),
#         batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
#
#     return train_loader, test_loader


def get_new_model():
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


FIX_MODEL_INIT = None


def get_model_assets(model_assets=None, reset_model=True, use_same_init=True):
    global FIX_MODEL_INIT
    if reset_model:
        if use_same_init and FIX_MODEL_INIT is not None:
            if FIX_MODEL_INIT is None:
                FIX_MODEL_INIT = get_new_model()
            return copy.deepcopy(FIX_MODEL_INIT)
        else:
            return get_new_model()
    if model_assets is None:
        return get_new_model()
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


def get_linear_probe_model(model_assets):
    model, optimizer = model_assets
    model.eval()

    def get_latent_fn(enc_model, x):
        mu, logvar = enc_model.encoder(x.view(-1, 1, 32, 32))
        return mu.reshape(mu.shape[0], -1)

    linear_probe_model = LinearProbeModel(model, input_dim=model.latent_dim, output_dim=10,
                                          get_latent_fn=get_latent_fn)
    return linear_probe_model.to(device)
