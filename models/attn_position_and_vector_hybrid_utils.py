import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


class SparseAttnBottleneck(nn.Module):
    def __init__(self, voc_size, dim, num_topk):
        super().__init__()
        self.voc_size = voc_size
        self.codebook = nn.Parameter(torch.randn(voc_size, dim))
        import math
        # initialize the codebook just like linear layer
        nn.init.kaiming_uniform_(self.codebook, a=math.sqrt(5))
        self.codebook.cuda()  # just a hack
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.num_topk = num_topk

    def max_neg_value(self, tensor):
        return -torch.finfo(tensor.dtype).max

    def forward(self, x):
        # x: (batch, *, dim)
        q = self.to_q(x)
        # here I first assume k and v are both comes from the codebook
        k = self.to_k(self.codebook)
        v = self.to_v(self.codebook)
        # attention
        dots = torch.einsum('bd,kd->bk', q, k)

        # top k of attention
        top, _ = dots.topk(self.num_topk, dim=-1)
        vk = top[..., -1].unsqueeze(-1).expand_as(dots)
        mask = dots < vk
        mask_value = self.max_neg_value(dots)
        dots.masked_fill_(mask, mask_value)

        # softmax
        attn = F.softmax(dots, dim=-1)

        # get output by applying attention weight to v
        out = torch.einsum('bk,kd->bd', attn, v)
        return out


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

        # use 1x1 conv to reduce the dimension of the feature map to 6 channels
        self.bottleneck_conv = nn.Conv2d(self.code_sz, 6, 1)
        self.bottleneck_fc = nn.Linear((image_size // 4) * (image_size // 4), latent_dim)

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
        # latent: (batch, dim， 7，7)
        position_feature = self.bottleneck_conv(latent)  # (batch, num_components, 7, 7)
        b, c, h, w = position_feature.shape
        position_feature_max_pos = torch.argmax(position_feature.reshape(b, c, h * w),
                                                dim=1)  # (batch, num_components)

        feature_vector = self.bottleneck_fc(position_feature.reshape(b*c, -1))  # (batch * num_components, hidden_dim)
        emb = self.embedder(feature_vector)
        emb_after_attn = self.sparse_attention(emb)
        emb_proj = self.projector(emb_after_attn)  # (batch * num_components, hidden_dim)

        feature_map_out = torch.zeros_like(latent).reshape(b, -1, h * w).transpose(1, 2)  # (batch, 7*7, hidden_dim)

        # fill the feature map with the projected embedding according to position_feature_max_pos
        for i in range(c):
            feature_map_out[torch.arange(b), position_feature_max_pos[:, i], :] += emb_proj[i * b: (i + 1) * b, :]

        return feature_map_out.reshape(b, -1, h, w)

    def encode(self, x):
        h = self.enc(x.view(-1, self.ch, self.image_size, self.image_size))
        return h

    def decode(self, z):
        return self.dec(z)

    def forward(self, x, **kwargs):
        z = self.encode(x)
        latent = self.bottleneck(z)
        out = self.decode(latent)
        return out


from torchvision import datasets, transforms
import utils.data_utils


def get_data_config(dataset_name):
    if dataset_name in ['mnist', 'fashion_mnist', 'kuzushiji', 'google_draw']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(28),
        ])
        config = {'image_size': 28, 'ch': 1, 'transform': transform, 'out_act': 'sigmoid'}
        return config
    elif dataset_name == 'omniglot':
        transform = transforms.Compose([
            transforms.ToTensor(),
            utils.data_utils.flip_image_value,
            transforms.Resize(28),
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
    elif dataset_name == 'bach':
        transform = None
        config = {'image_size': 48, 'ch': 1, 'transform': transform, 'out_act': 'sigmoid'}
        return config
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_new_model(**kwargs):
    model = SparseAttnAutoencoderFeatureMap(**kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer
