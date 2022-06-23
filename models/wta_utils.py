import torch
import torch.utils.data
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import copy

import utils.data_utils
from evaluate.linear_probe import LinearProbeModel
from utils.iter_recon_utils import recon_till_converge
from .vqvae_utils import ResBlock

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 100
NUM_WORKERS = 8
DIFF_THRES = 1e-6
MAX_RECON_ITER = 100
TOTAL_EPOCH = 50  # maybe 100


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.normal_(0.0, 0.001)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def equals(x, y, eps=1e-8):
    return torch.abs(x - y) <= eps


# courtesy of Mehdi C
# https://github.com/mehdidc/ae_gen
def spatial_sparsity(x):
    maxes = x.amax(dim=(2, 3), keepdims=True)
    return x * equals(x, maxes)


def lifetime_sparsity(h, rate=0.05):
    shp = h.shape
    n = shp[0]
    c = shp[1]
    h_reshape = h.reshape((n, -1))
    thr, ind = torch.topk(h_reshape, int(rate * n), dim=0)
    batch_mask = 0. * h_reshape
    batch_mask.scatter_(0, ind, 1)
    batch_mask = batch_mask.reshape(shp)
    return h * batch_mask


def channel_sparsity(h, rate=0.05):
    shp = h.shape
    n = shp[0]
    c = shp[1]
    h_reshape = h.reshape((n, c, -1))
    thr, ind = torch.topk(h_reshape, int(rate * c), dim=1)
    batch_mask = 0. * h_reshape
    batch_mask.scatter_(1, ind, 1)
    batch_mask = batch_mask.reshape(shp)
    return h * batch_mask


class WTA(nn.Module):
    # https://github.com/iwyoo/tf_ConvWTA/blob/master/model.py
    def __init__(self, sz=64, code_sz=128,
                 lifetime_sparsity_rate=0.05,
                 channel_sparsity_rate=1, ch=1,
                 image_size=28, sample_num=1024,
                 kernel_size=5, net_type='wta',
                 out_act='sigmoid',
                 **kwargs):
        super(WTA, self).__init__()
        self.sz = sz
        self.code_sz = code_sz
        self.lifetime_sparsity_rate = lifetime_sparsity_rate
        self.channel_sparsity_rate = channel_sparsity_rate
        self.ch = ch
        self.image_size = image_size
        self.sample_z = torch.rand(sample_num, ch, image_size, image_size)

        if net_type == 'wta':
            self.enc = nn.Sequential(
                # input is Z, going into a convolution
                nn.Conv2d(ch, sz, kernel_size, 1, padding=0),
                nn.ReLU(True),
                nn.Conv2d(sz, sz, kernel_size, 1, padding=0),
                nn.ReLU(True),
                nn.Conv2d(sz, self.code_sz, kernel_size, 1, padding=0),
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
        self.apply(weights_init)

    def encode(self, x):
        h = self.enc(x.view(-1, self.ch, self.image_size, self.image_size))
        return h

    def decode(self, z):
        return self.dec(z)

    def forward(self, x, spatial=True, lifetime=True):
        z = self.encode(x)
        if spatial:
            z = spatial_sparsity(z)
        if self.channel_sparsity_rate < 1:
            z = channel_sparsity(z)
        if lifetime:
            z = lifetime_sparsity(z, self.lifetime_sparsity_rate)
        out = self.decode(z)
        return out


def train_one_epoch(model, optimizer, train_data):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_data):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = .5 * ((recon_batch - data) ** 2).sum() / data.shape[0]  # loss_function(recon_batch, data)
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


def evaluate(model_assets, test_data):
    model, optimizer = model_assets
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_data):
            data = data.to(device)
            recon_batch = model(data)
            loss = .5 * ((recon_batch - data) ** 2).sum() / data.shape[0]  # loss_function(recon_batch, data)
            test_loss += loss.item()

        print('====> Average test loss: {:.4f}'.format(
            test_loss / len(test_data)))
    return model, optimizer


def train_with_teacher(new_model_assets, old_model_assets, steps, **kwargs):
    model, optimizer = new_model_assets
    model.train()
    train_loss = []
    for batch_idx in range(steps):
        data = gen_data(old_model_assets, BATCH_SIZE, 1).to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = .5 * ((recon_batch - data) ** 2).sum() / data.shape[0]
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        if batch_idx % 100 == 0:
            print('====> Step: {} Average loss: {:.4f}'.format(
                batch_idx, np.mean(train_loss)))

    return model, optimizer


def get_transform(dataset_name):
    if dataset_name in ['mnist', 'fashion-mnist', 'kuzushiji']:
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    elif dataset_name == 'omniglot':
        return transforms.Compose([
            transforms.ToTensor(),
            utils.data_utils.flip_image_value,
            transforms.Resize(28),
        ])
    elif dataset_name in ['cifar10', 'cifar100']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif dataset_name == 'wikiart':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((64, 64)),  # TODO
        ])
    elif dataset_name in ['mpi3d', 'dsprite']:
        return None

    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))


def get_new_model(**kwargs):
    model = WTA(**kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


FIX_MODEL_INIT = None


def get_model_assets(model_assets=None, reset_model=True, use_same_init=True, **kwargs):
    global FIX_MODEL_INIT
    if reset_model:
        if use_same_init and FIX_MODEL_INIT is not None:
            if FIX_MODEL_INIT is None:
                FIX_MODEL_INIT = get_new_model(**kwargs)
            return copy.deepcopy(FIX_MODEL_INIT)
        else:
            return get_new_model(**kwargs)
    if model_assets is None:
        return get_new_model(**kwargs)
    else:
        return model_assets


def get_train_data_next_iter(train_data, data_generated, add_old_dataset=False, keep_portion=1.0):
    return utils.data_utils.get_train_data_next_iter(train_data, data_generated, add_old_dataset=add_old_dataset,
                                                     keep_portion=keep_portion, batch_size=BATCH_SIZE,
                                                     num_workers=NUM_WORKERS)


def recon_fn(model, image_batch):
    return model(image_batch, spatial=True, lifetime=False)


def gen_data(model_assets, gen_batch_size, gen_num_batch, thres=DIFF_THRES, max_iteration=MAX_RECON_ITER, **kwargs):
    data_all = []
    model, optimizer = model_assets
    model.eval()
    ch = model.ch
    image_size = model.image_size
    for _ in range(gen_num_batch):
        noise = torch.rand((gen_batch_size, ch, image_size, image_size)).to(device)
        sample = recon_till_converge(model, recon_fn, noise, thres=thres, max_iteration=max_iteration).cpu()
        data_all.append(sample)
    data_all = torch.cat(data_all, dim=0)
    return data_all


def get_kernel_visualization(model):
    code_size = model.code_sz
    image_size = model.image_size
    latent = torch.zeros([code_size, code_size, image_size - 12, image_size - 12])  # See page 167 of the PHD thesis
    for i in range(code_size):
        latent[i, i, (image_size - 12) // 2, (image_size - 12) // 2] = 1  # Set middle point as 1
    with torch.no_grad():
        latent = latent.to(device).float()
        img = model.dec(latent)
    return img


def save_sample(model_assets, log_dir, iteration, thres=DIFF_THRES, max_iteration=MAX_RECON_ITER):
    model, optimizer = model_assets
    model.eval()
    sample_z = model.sample_z.to(device)
    image_size = model.image_size
    ch = model.ch
    sample_num = sample_z.shape[0]
    sample = recon_till_converge(model, recon_fn, sample_z, thres=thres, max_iteration=max_iteration).cpu()
    save_image(sample.view(sample_num, ch, image_size, image_size), f'{log_dir}/sample_iter_{iteration}_full' + '.png',
               nrow=32)
    save_image(sample.view(sample_num, ch, image_size, image_size)[:64],
               f'{log_dir}/sample_iter_{iteration}_small' + '.png', nrow=8)
    kernel_img = get_kernel_visualization(model)
    save_image(kernel_img.view(model.code_sz, ch, image_size, image_size),
               f'{log_dir}/kernel_iter_{iteration}' + '.png', nrow=8)


def get_linear_probe_model(model_assets):
    model, optimizer = model_assets
    model.eval()
    image_size = model.image_size

    def get_latent_fn(enc_model, x):
        x = enc_model.enc(x)
        x = enc_model.spatial_sparsity(x)
        return x.reshape(x.shape[0], -1)

    linear_probe_model = LinearProbeModel(model, input_dim=model.code_sz * (image_size - 12) * (image_size - 12),
                                          output_dim=10,  # TODO: change this to the number of classes
                                          get_latent_fn=get_latent_fn)
    return linear_probe_model.to(device)
