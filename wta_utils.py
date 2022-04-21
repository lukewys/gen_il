import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import copy


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.normal_(0.0, 0.001)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 100
NUM_WORKERS = 8
SAMPLE_NUM = 1024
SAMPLE_Z = torch.rand(SAMPLE_NUM, 1, 28, 28).to(device)
DIFF_THRES = 1e-6
MAX_RECON_ITER = 100


class WTA(nn.Module):
    # https://github.com/iwyoo/tf_ConvWTA/blob/master/model.py
    def __init__(self):
        super(WTA, self).__init__()
        sz = 128

        self.enc = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(1, sz, 5, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(sz, sz, 5, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(sz, sz, 5, 1, 0),
            nn.ReLU(True),
        )
        self.sig = nn.Sigmoid()
        self.dec = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(sz, sz, 5, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(sz, sz, 5, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(sz, 1, 5, 1, 0)
        )
        # self.internal_random_state = torch.Generator()
        # self.internal_random_state.manual_seed(12345)
        # torch.rand(5, generator=gen0)

    def encode(self, x):
        h = self.enc(x.view(-1, 1, 28, 28))
        return h

    def decode(self, z):
        return self.dec(z)

    def spatial_sparsity(self, h):
        n = h.shape[0]
        c = h.shape[1]
        # h_reshape = h.reshape(n,c,-1)
        # thr, ind = torch.topk(h_reshape, 1, dim=-1)
        mask = torch.where(h == h.max(axis=3)[0].max(axis=2)[0][..., None, None], torch.ones(1).to(device),
                           torch.zeros(1).to(device))
        return h * mask

    def lifetime_sparsity(self, h, rate=0.05):
        shp = h.shape
        n = shp[0]
        c = shp[1]
        h_reshape = h.reshape((n, -1))
        thr, ind = torch.topk(h_reshape, int(rate * n), dim=0)
        batch_mask = 0. * h_reshape
        batch_mask.scatter_(0, ind, 1)
        batch_mask = batch_mask.reshape(shp)
        return h * batch_mask

    def forward(self, x, spatial=True, lifetime=True):
        z = self.encode(x)
        if spatial:
            z = self.spatial_sparsity(z)
        if lifetime:
            z = self.lifetime_sparsity(z)
        out = self.decode(z)
        return out


def train(model_assets, train_data, train_extend):
    model, optimizer = model_assets
    model.train()
    total_epoch = 50  # maybe 100
    total_epoch = int(np.round(train_extend * total_epoch))
    for epoch in range(total_epoch):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_data):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch = model(data)
            loss = .5 * ((recon_batch - data) ** 2).sum() / BATCH_SIZE  # loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_data)))
    return model, optimizer


def train_with_teacher(new_model_assets, old_model_assets, steps):
    model, optimizer = new_model_assets
    model.train()
    train_loss = []
    for batch_idx in range(steps):
        data = gen_data(old_model_assets, BATCH_SIZE, 1).to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = .5 * ((recon_batch - data) ** 2).sum() / BATCH_SIZE
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
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist_data/', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader


def get_new_model():
    model = WTA().to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


FIX_MODEL_INIT = get_new_model()


def get_model_assets(model_assets=None, reset_model=True, use_same_init=True):
    if reset_model:
        if use_same_init:
            return copy.deepcopy(FIX_MODEL_INIT)
        else:
            return get_new_model()
    else:
        return model_assets


def get_train_data_next_iter(train_data, data_generated, add_old_dataset=False, batch_size=BATCH_SIZE):
    if add_old_dataset:
        old_data_all = []
        for batch_idx, (data, _) in enumerate(train_data):
            old_data_all.append(data)
        old_data_all = torch.cat(old_data_all, dim=0)
        data_combined = torch.cat([old_data_all.view(-1, 1, 28, 28), data_generated], dim=0)
        train_loader_new = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_combined, data_combined),
            batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        return train_loader_new
    else:
        train_loader_new = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_generated, data_generated),
            batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader_new


def renormalize(x):
    # x is a tensor of shape (batch_size, 28, 28)
    x = x.view(-1, 28 * 28)
    x = (x - torch.min(x, dim=1, keepdim=True)[0]) / (
            torch.max(x, dim=1, keepdim=True)[0] - torch.min(x, dim=1, keepdim=True)[0])
    return x.view(-1, 28, 28)


def recon_till_converge(model, image_batch, thres=1e-6, return_history=False, max_iteration=100):
    diff = 1e10
    history = [image_batch]
    model.eval()
    iteration = 1
    with torch.no_grad():
        while diff > thres:
            recon_batch = model(image_batch, lifetime=False, spatial=True)
            diff = torch.max(torch.mean((image_batch - recon_batch).pow(2), dim=(1, 2)))
            if return_history:
                history.append(recon_batch)
            # re-normalize image
            image_batch = renormalize(recon_batch)
            iteration += 1
            if iteration > max_iteration:
                break
    if return_history:
        return recon_batch, history
    else:
        return recon_batch


def gen_data(model_assets, gen_batch_size, gen_num_batch):
    data_all = []
    model, optimizer = model_assets
    model.eval()
    for _ in range(gen_num_batch):
        noise = torch.rand((gen_batch_size, 28, 28)).to(device)
        sample = recon_till_converge(model, noise, thres=DIFF_THRES, max_iteration=MAX_RECON_ITER).cpu()
        data_all.append(sample)
    data_all = torch.cat(data_all, dim=0)
    return data_all


def save_sample(model_assets, log_dir, iteration):
    model, optimizer = model_assets
    sample = recon_till_converge(model, SAMPLE_Z, thres=DIFF_THRES, max_iteration=MAX_RECON_ITER).cpu()
    save_image(sample.view(SAMPLE_NUM, 1, 28, 28), f'{log_dir}/sample_iter_{iteration}_full' + '.png', nrow=32)
    save_image(sample.view(SAMPLE_NUM, 1, 28, 28)[:64], f'{log_dir}/sample_iter_{iteration}_small' + '.png', nrow=8)
