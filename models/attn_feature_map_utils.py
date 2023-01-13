import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


from models.sparse_attention_bottleneck import SparseAttnBottleneck
from models.wta_utils import ResBlock, weights_init_wta, weights_init_vqvae


class SparseAttnAutoencoderFeatureMap(nn.Module):
    # https://github.com/iwyoo/tf_ConvWTA/blob/master/model.py
    def __init__(self, latent_dim=128, hid_channels=64, voc_size=8, num_topk=6, proj_hidden_dim=512,
                 ch=1, image_size=28, sample_num=1024,
                 kernel_size=5, net_type='vqvae',
                 out_act='sigmoid',
                 **kwargs):
        super(SparseAttnAutoencoderFeatureMap, self).__init__()
        sz = hid_channels
        self.sz = sz
        latent_dim = hid_channels
        code_sz = latent_dim
        self.hid_channels = hid_channels
        self.latent_dim = latent_dim
        self.proj_output_dim = latent_dim
        self.code_sz = code_sz
        self.ch = ch
        self.image_size = image_size
        self.sample_z = torch.rand(sample_num, ch, image_size, image_size)

        if net_type == 'wta':
            self.enc = nn.Sequential(
                # input is Z, going into a convolution
                nn.Conv2d(ch, sz, kernel_size, 1, 0),
                nn.ReLU(True),
                nn.Conv2d(sz, sz, kernel_size, 1, 0),
                nn.ReLU(True),
                nn.Conv2d(sz, self.code_sz, kernel_size, 1, 0),
                nn.ReLU(True),
            )
            self.dec = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(self.code_sz, sz, kernel_size, 1, 0),
                nn.ReLU(True),
                nn.ConvTranspose2d(sz, sz, kernel_size, 1, 0),
                nn.ReLU(True),
                nn.ConvTranspose2d(sz, ch, kernel_size, 1, 0),
            )
        elif net_type == 'vqvae':
            dim = sz
            self.code_sz = sz
            self.enc = nn.Sequential(
                nn.Conv2d(ch, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 4, 2, 1),
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
                nn.ConvTranspose2d(dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, ch, 4, 2, 1),
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
        self.embedder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim, bias=False),
            nn.BatchNorm1d(self.latent_dim)
        )
        self.sparse_attention = SparseAttnBottleneck(voc_size, self.latent_dim, num_topk=num_topk)

        self.projector = nn.Sequential(
            nn.Linear(self.latent_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, self.proj_output_dim),
        )
        if net_type == 'wta':
            self.apply(weights_init_wta)
        elif net_type == 'vqvae':
            self.apply(weights_init_vqvae)
        self.net_type = net_type

    def bottleneck(self, latent):
        emb = self.embedder(latent)
        emb_after_attn, top_value, top_ind, dots = self.sparse_attention(emb)
        emb_proj = self.projector(emb_after_attn)
        return emb_proj

    def encode(self, x):
        h = self.enc(x.view(-1, self.ch, self.image_size, self.image_size))
        return h

    def decode(self, z):
        return self.dec(z)

    def forward(self, x, **kwargs):
        z = self.encode(x)
        z_shape = z.shape
        z = z.permute(0, 2, 3, 1).contiguous().view(-1, self.code_sz)
        latent = self.bottleneck(z)
        latent = latent.reshape(z_shape)
        out = self.decode(latent)
        return out


from torchvision import datasets, transforms
import utils.data_utils


def get_data_config(dataset_name):
    if dataset_name in ['mnist', 'fashion_mnist', 'kuzushiji', 'google_draw']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
        ])
        config = {'image_size': 32, 'ch': 1, 'transform': transform, 'out_act': 'sigmoid'}
        return config
    elif dataset_name == 'omniglot':
        transform = transforms.Compose([
            transforms.ToTensor(),
            utils.data_utils.flip_image_value,
            transforms.Resize(32),
        ])
        config = {'image_size': 32, 'ch': 1, 'transform': transform, 'out_act': 'sigmoid'}
        return config
    elif dataset_name in ['cifar10', 'cifar100']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        config = {'image_size': 32, 'ch': 3, 'transform': transform, 'out_act': 'tanh'}
        return config
    elif dataset_name in ['wikiart', 'celeba']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((64, 64)),  # TODO: decide image size
        ])
        config = {'image_size': 64, 'ch': 3, 'transform': transform, 'out_act': 'tanh'}
        return config
    elif dataset_name in ['mpi3d', 'dsprite']:
        transform = None
        config = {'image_size': 64, 'ch': 1, 'transform': transform, 'out_act': 'sigmoid'}
        return config
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_new_model(**kwargs):
    model = SparseAttnAutoencoderFeatureMap(**kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer
