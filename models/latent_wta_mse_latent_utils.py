import torch
import torch.utils.data
from torch import nn, optim
from torchvision import transforms
import numpy as np
import copy


from models.wta_utils import ResBlock, weights_init_wta, weights_init_vqvae, spatial_sparsity, lifetime_sparsity, \
    channel_sparsity, TOTAL_EPOCH, evaluate, save_sample

from models.pretrain_vqvae import vq_st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LatentWTA(nn.Module):
    def __init__(self, vqvae=None, sz=128, code_sz=128,
                 lifetime_sparsity_rate=0.05,
                 channel_sparsity_rate=1, image_ch=1,
                 wta_in_ch=256, wta_out_ch=256,
                 image_size=28, sample_num=1024,
                 kernel_size=5, net_type='wta',
                 out_act='none', loss_fn='mse',
                 **kwargs):
        super(LatentWTA, self).__init__()
        self.sz = sz
        self.code_sz = code_sz
        self.lifetime_sparsity_rate = lifetime_sparsity_rate
        self.channel_sparsity_rate = channel_sparsity_rate
        self.image_ch = image_ch
        self.image_size = image_size
        self.sample_z = torch.rand(sample_num, image_ch, image_size, image_size)
        self.loss_fn = loss_fn

        if net_type == 'wta':
            self.enc = nn.Sequential(
                # input is Z, going into a convolution
                nn.Conv2d(wta_in_ch, sz, kernel_size, 1, 'same'),
                nn.ReLU(True),
                nn.Conv2d(sz, sz, kernel_size, 1, 'same'),
                nn.ReLU(True),
                nn.Conv2d(sz, self.code_sz, kernel_size, 1, 'same'),
                nn.ReLU(True),
            )
            self.dec = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(self.code_sz, sz, kernel_size, 1, 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(sz, sz, kernel_size, 1, 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(sz, wta_out_ch, kernel_size, 1, 2),
            )
        elif net_type == 'vqvae':
            dim = sz
            self.code_sz = sz
            self.enc = nn.Sequential(
                nn.Conv2d(wta_in_ch, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 4, 2, 1),
                ResBlock(dim),
                ResBlock(dim),
            )

            self.dec = nn.Sequential(
                ResBlock(dim),
                ResBlock(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, wta_out_ch, 4, 2, 1),
            )
        if out_act == 'sigmoid':
            self.dec = nn.Sequential(self.dec, nn.Sigmoid())
        elif out_act == 'tanh':
            self.dec = nn.Sequential(self.dec, nn.Tanh())
        elif out_act == 'none':
            pass
        else:
            raise ValueError('Unknown output activation function: {}'.format(out_act))
        self.out_act = out_act
        if net_type == 'wta':
            self.apply(weights_init_wta)
        elif net_type == 'vqvae':
            self.apply(weights_init_vqvae)
        self.net_type = net_type

        self.vqvae = vqvae
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def encode(self, x):
        h = self.enc(x)
        return h

    def decode(self, z):
        return self.dec(z)

    def vq_encode(self, x):
        z_e_x = self.vqvae.encoder(x)
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.vqvae.codebook.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()
        return z_q_x, indices

    def vq_decode(self, z_q_x):
        return self.vqvae.decoder(z_q_x)

    def forward(self, x, spatial=True, lifetime=True):
        z_q_x_gt, indices_gt = self.vq_encode(x)
        z = self.encode(z_q_x_gt)
        if spatial:
            z = spatial_sparsity(z)
        if self.channel_sparsity_rate < 1:
            z = channel_sparsity(z, rate=self.channel_sparsity_rate)
        if lifetime:
            z = lifetime_sparsity(z, rate=self.lifetime_sparsity_rate)
        z_q_x_pred = self.decode(z)
        out = self.vq_decode(z_q_x_pred)
        if self.training:
            return out, z_q_x_gt, z_q_x_pred
        else:
            return out


def train_one_epoch(model, optimizer, train_data):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_data):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, z_q_x_gt, z_q_x_pred = model(data)
        loss = nn.MSELoss()(z_q_x_pred, z_q_x_gt)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return model, optimizer, train_loss / len(train_data)


def train(model_assets, train_data, train_extend):
    model, optimizer = model_assets
    model.train()
    total_epoch = int(np.round(train_extend * TOTAL_EPOCH))
    for epoch in range(total_epoch):
        model, optimizer, training_loss = train_one_epoch(model, optimizer, train_data)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, training_loss))
    return model, optimizer



def get_data_config(dataset_name):
    if dataset_name in ['mnist', 'fashion_mnist', 'kuzushiji', 'google_draw']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        config = {'image_size': 28, 'ch': 1, 'transform': transform, 'out_act': 'sigmoid'}
        return config
    elif dataset_name in ['cifar10', 'cifar100']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        config = {'image_size': 32, 'ch': 3, 'transform': transform, 'out_act': 'tanh'}
        return config
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))


def get_transform(dataset_name):
    if dataset_name in ['mnist']:
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    elif dataset_name in ['cifar10', 'cifar100']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))

def get_new_model(**kwargs):
    model = LatentWTA(**kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


FIX_MODEL_INIT = None


def get_model_assets(model_assets=None, reset_model=True, use_same_init=True, **kwargs):
    global FIX_MODEL_INIT
    if reset_model:
        if FIX_MODEL_INIT is None and use_same_init:
            FIX_MODEL_INIT = get_new_model(**kwargs)
        if use_same_init:
            return copy.deepcopy(FIX_MODEL_INIT)
        else:
            return get_new_model(**kwargs)
    elif model_assets is None:
        return get_new_model(**kwargs)
    else:
        return model_assets