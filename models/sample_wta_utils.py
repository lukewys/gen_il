import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import copy

import utils.data_utils
from evaluate.linear_probe import LinearProbeModel
from utils.iter_recon_utils import recon_till_converge
from utils.music_data_utils import single_channel_pianoroll_save_to_midi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 100
NUM_WORKERS = 8
DIFF_THRES = 1e-3
MAX_RECON_ITER = 100
TOTAL_EPOCH = 50  # maybe 100


# TODO: make these parameters as properties in a class


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


def weights_init_wta(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.normal_(0.0, 0.001)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_vqvae(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


def equals(x, y, eps=1e-8):
    return torch.abs(x - y) <= eps


# courtesy of Mehdi C
# https://github.com/mehdidc/ae_gen
def spatial_sparsity(x):
    maxes = x.amax(dim=(2, 3), keepdims=True)
    return x * equals(x, maxes)


def spatial_sparsity_sample(x, sample=True):
    inp = x
    if not sample:
        return spatial_sparsity(x)
    ori_shp = x.shape
    x = x.reshape((x.shape[0] * x.shape[1], -1))
    p = F.softmax(x, dim=1)
    sample = torch.multinomial(p, 1)
    mask = torch.zeros_like(x)
    mask.scatter_(-1, sample, 1)
    mask = mask.reshape(ori_shp)
    return inp * mask


def lifetime_sparsity(h, rate=0.05):
    if rate == 1:
        return h
    shp = h.shape
    n = shp[0]
    c = shp[1]
    h_reshape = h.sum((2, 3)).reshape((n, -1))
    thr, ind = torch.topk(h_reshape, int(rate * n), dim=0)
    batch_mask = 0. * h_reshape
    # set the top k channels to 1
    batch_mask[ind, torch.arange(c)] = 1
    return h * batch_mask[:, :, None, None]


def vector_sparsity(h, rate=0.5):
    # h: (n, c), take the top k channels
    if rate == 1:
        return h
    shp = h.shape
    n = shp[0]
    c = shp[1]
    thr, ind = torch.topk(h, int(rate * c), dim=1)
    batch_mask = 0. * h
    batch_mask.scatter_(1, ind, 1)
    return h * batch_mask


# x = torch.zeros(100, 128, 16, 16)
# for b in range(100):
#     for i in range(128):
#         x[b][i][2][2] = torch.rand(1)
#
# print((x != 0).sum())
# s = lifetime_sparsity(x, 0.05)
# print((s != 0).sum())
# print((s[:,0] != 0).sum())


def channel_sparsity(h, rate=0.05):
    if rate == 1:
        return h
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
    def __init__(self, sz=128, code_sz=128,
                 lifetime_sparsity_rate=0.05,
                 channel_sparsity_rate=1, ch=1,
                 image_size=28, sample_num=1024,
                 kernel_size=5, net_type='wta',
                 out_act='sigmoid', loss_fn='mse',
                 denoise='none', noise_factor=0.2,
                 **kwargs):
        super(WTA, self).__init__()
        self.sz = sz
        self.code_sz = code_sz
        self.lifetime_sparsity_rate = lifetime_sparsity_rate
        self.channel_sparsity_rate = channel_sparsity_rate
        self.ch = ch
        self.image_size = image_size
        self.sample_z = torch.rand(sample_num, ch, image_size, image_size)
        self.loss_fn = loss_fn
        self.denoise = denoise
        self.noise_factor = noise_factor
        if out_act == 'tanh' or self.ch == 3:
            self.sample_z = self.sample_z * 2 - 1  # [-1, 1]

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
            self.enc = nn.Sequential(
                nn.Conv2d(ch, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, self.code_sz, 4, 2, 1),
                ResBlock(self.code_sz),
                ResBlock(self.code_sz),
            )

            self.dec = nn.Sequential(
                ResBlock(self.code_sz),
                ResBlock(self.code_sz),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.code_sz, dim, 4, 2, 1),
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
        if net_type == 'wta':
            self.apply(weights_init_wta)
        elif net_type == 'vqvae':
            self.apply(weights_init_vqvae)
        self.net_type = net_type

    def encode(self, x):
        h = self.enc(x.view(-1, self.ch, self.image_size, self.image_size))
        return h

    def decode(self, z):
        return self.dec(z)

    def forward(self, x, spatial=True, lifetime=True):
        z = self.encode(x)
        if spatial:
            z = spatial_sparsity_sample(z, self.training)
        if self.channel_sparsity_rate < 1:
            z = channel_sparsity(z, rate=self.channel_sparsity_rate)
        if lifetime:
            z = lifetime_sparsity(z, rate=self.lifetime_sparsity_rate)
        out = self.decode(z)
        return out


def train_one_epoch(model, optimizer, train_data):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_data):
        data = data.to(device)
        if hasattr(model, 'denoise'):
            if model.denoise == 'gaussian':
                data_inp = data + torch.randn_like(data) * model.noise_factor
            elif model.denoise == 'uniform':
                data_inp = data + torch.rand_like(data) * model.noise_factor
            elif model.denoise == 'gaussian-clip':
                data_inp = torch.clip(data + torch.randn_like(data) * model.noise_factor, 0, 1)
            elif model.denoise == 'uniform-clip':
                data_inp = torch.clip(data + torch.rand_like(data) * model.noise_factor, 0, 1)
            else:
                data_inp = data
        else:
            data_inp = data
        optimizer.zero_grad()
        recon_batch = model(data_inp)
        if not hasattr(model, 'loss_fn') or model.loss_fn == 'mse':
            loss = .5 * ((recon_batch - data) ** 2).sum() / data.shape[0]
        elif model.loss_fn == 'bce':
            loss = nn.BCELoss()(recon_batch, data)
        else:
            raise ValueError('Unknown loss function: {}'.format(model.loss_fn))
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


def evaluate(model_assets, test_data, transform, **kwargs):
    model, optimizer = model_assets
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_data):
            data = data.to(device)
            recon_batch = model(data)
            if not hasattr(model, 'loss_fn') or model.loss_fn == 'mse':
                loss = .5 * ((recon_batch - data) ** 2).sum() / data.shape[0]
            elif model.loss_fn == 'bce':
                loss = nn.BCELoss()(recon_batch, data)
            else:
                raise ValueError('Unknown loss function: {}'.format(model.loss_fn))
            test_loss += loss.item()

        print('====> Average test loss: {:.4f}'.format(
            test_loss / len(test_data)))

        data, _ = next(iter(test_data))
        data = data.to(device)
        recon = model(data)
        recon = recon.cpu().detach()
        image_size = model.image_size
        ch = data.shape[1]
        log_dir = kwargs['log_dir']
        iteration = kwargs['iteration']
        recon = utils.data_utils.denormalize(recon, transform)
        data = utils.data_utils.denormalize(data, transform)
        save_image(recon.view(recon.shape[0], ch, image_size, image_size)[:64],
                   f'{log_dir}/recon_iter_{iteration}' + '.png', nrow=8)
        save_image(data.view(data.shape[0], ch, image_size, image_size)[:64],
                   f'{log_dir}/test_gt_iter_{iteration}' + '.png', nrow=8)
    return model, optimizer


def train_with_teacher(new_model_assets, old_model_assets, steps, batch_size=BATCH_SIZE, **kwargs):
    model, optimizer = new_model_assets
    model.train()
    train_loss = []
    for batch_idx in range(steps):
        data = gen_data(old_model_assets, batch_size, 1, **kwargs).to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        if not hasattr(model, 'loss_fn') or model.loss_fn == 'mse':
            loss = .5 * ((recon_batch - data) ** 2).sum() / data.shape[0]
        elif model.loss_fn == 'bce':
            loss = nn.BCELoss()(recon_batch, data)
        else:
            raise ValueError('Unknown loss function: {}'.format(model.loss_fn))
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        if batch_idx % 100 == 0:
            print('====> Step: {} Average loss: {:.4f}'.format(
                batch_idx, np.mean(train_loss)))

    return model, optimizer


def get_data_config(dataset_name):
    if dataset_name in ['mnist', 'fashion_mnist', 'kuzushiji', 'google_draw']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(28),
        ])
        config = {'image_size': 28, 'ch': 1, 'transform': transform, 'out_act': 'none'}
        return config
    elif dataset_name == 'mnist-tanh':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(28),
            transforms.Normalize((0.5,), (0.5,))
        ])
        config = {'image_size': 28, 'ch': 1, 'transform': transform, 'out_act': 'tanh'}
        return config
    elif dataset_name == 'omniglot':
        transform = transforms.Compose([
            transforms.ToTensor(),
            utils.data_utils.flip_image_value,
            transforms.Resize(28),
        ])
        config = {'image_size': 28, 'ch': 1, 'transform': transform, 'out_act': 'none'}
        return config
    elif dataset_name in ['cifar10', 'cifar100']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        config = {'image_size': 32, 'ch': 3, 'transform': transform, 'out_act': 'tanh'}
        return config
    elif dataset_name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
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
    elif dataset_name in ['colored_mnist',
                          'colored_mnist_by_digits',
                          'colored_mnist_all_red',
                          'colored_mnist_all_dark_purple',
                          'colored_mnist_dark_purple_and_yellow']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])  # this is a hack. the transform will not be applied to the dataset
        config = {'image_size': 28, 'ch': 3, 'transform': transform, 'out_act': 'tanh'}
        return config
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))


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


def get_train_data_next_iter(train_data, data_generated, add_old_dataset=False, keep_portion=1.0,
                             batch_size=BATCH_SIZE):
    return utils.data_utils.get_train_data_next_iter(train_data, data_generated, add_old_dataset=add_old_dataset,
                                                     keep_portion=keep_portion, batch_size=batch_size,
                                                     num_workers=NUM_WORKERS)


def recon_fn(model, image_batch):
    return model(image_batch, spatial=True, lifetime=False)


def recon_fn_lifetime(model, image_batch):
    return model(image_batch, spatial=True, lifetime=True)


def gen_data(model_assets, gen_batch_size, gen_num_batch, thres=DIFF_THRES, max_iteration=MAX_RECON_ITER, **kwargs):
    data_all = []
    model, optimizer = model_assets
    model.eval()
    ch = model.ch
    image_size = model.image_size
    for _ in range(gen_num_batch):
        noise = torch.rand((gen_batch_size, ch, image_size, image_size)).to(device)
        sample = recon_till_converge(model, recon_fn, noise, thres=thres, max_iteration=max_iteration,
                                     renorm=kwargs['renorm'], no_renorm_last_iter=kwargs['no_renorm_last_iter']).cpu()
        data_all.append(sample)
    data_all = torch.cat(data_all, dim=0)
    return data_all


def get_kernel_visualization(model, act=1):
    code_size = model.code_sz
    image_size = model.image_size
    if model.net_type == 'wta':
        latent = torch.zeros([code_size, code_size, image_size - 12, image_size - 12])  # See page 167 of the PHD thesis
        for i in range(code_size):
            latent[i, i, (image_size - 12) // 2, (image_size - 12) // 2] = act  # Set middle point as 1
    elif model.net_type == 'vqvae':
        latent = torch.zeros([code_size, code_size, image_size // 4, image_size // 4])
        for i in range(code_size):
            latent[i, i, (image_size // 4) // 2, (image_size // 4) // 2] = act  # Set middle point as 1
    elif model.net_type == 'vqvae-0-down':
        if hasattr(model, 'vqvae'):
            image_size = image_size // 4
        latent = torch.zeros([code_size, code_size, image_size, image_size])
        for i in range(code_size):
            latent[i, i, (image_size) // 2, (image_size) // 2] = act  # Set middle point as 1
    elif model.net_type == 'vqvae-1-down':
        if hasattr(model, 'vqvae'):
            image_size = image_size // 4
        latent = torch.zeros([code_size, code_size, image_size // 2, image_size // 2])
        for i in range(code_size):
            latent[i, i, (image_size // 2) // 2, (image_size // 2) // 2] = act
    else:
        raise ValueError('Unknown net type: {}'.format(model.net_type))
    with torch.no_grad():
        latent = latent.to(device).float()
        img = model.decode(latent)
    return img


def save_sample(model_assets, log_dir, iteration, transform,
                thres=DIFF_THRES, max_iteration=MAX_RECON_ITER, save_kernel=True,
                **gen_kwargs):
    model, optimizer = model_assets
    model.eval()
    sample_z = model.sample_z.to(device)
    image_size = model.image_size
    ch = sample_z.shape[1]
    sample_num = sample_z.shape[0]
    sample = recon_till_converge(model, recon_fn, sample_z, thres=thres, max_iteration=max_iteration,
                                 renorm=gen_kwargs['renorm'],
                                 no_renorm_last_iter=gen_kwargs['no_renorm_last_iter']).cpu()
    sample = utils.data_utils.denormalize(sample, transform)
    save_image(sample.view(sample_num, ch, image_size, image_size), f'{log_dir}/sample_iter_{iteration}_full' + '.png',
               nrow=32)
    save_image(sample.view(sample_num, ch, image_size, image_size)[:64],
               f'{log_dir}/sample_iter_{iteration}_small' + '.png', nrow=8)
    # save image as numpy array
    np.save(f'{log_dir}/sample_iter_{iteration}_full' + '.npy',
            sample.view(sample_num, ch, image_size, image_size).numpy())
    if save_kernel:
        kernel_img = get_kernel_visualization(model, act=1)
        kernel_img = utils.data_utils.denormalize(kernel_img, transform)
        save_image(kernel_img.view(model.code_sz, ch, image_size, image_size),
                   f'{log_dir}/kernel_iter_{iteration}_act_1' + '.png', nrow=8)
        kernel_img = get_kernel_visualization(model, act=2)
        kernel_img = utils.data_utils.denormalize(kernel_img, transform)
        save_image(kernel_img.view(model.code_sz, ch, image_size, image_size),
                   f'{log_dir}/kernel_iter_{iteration}_act_2' + '.png', nrow=8)
        kernel_img = get_kernel_visualization(model, act=4)
        kernel_img = utils.data_utils.denormalize(kernel_img, transform)
        save_image(kernel_img.view(model.code_sz, ch, image_size, image_size),
                   f'{log_dir}/kernel_iter_{iteration}_act_4' + '.png', nrow=8)
    if gen_kwargs['dataset_name'] == 'bach':
        sample = sample.detach().cpu().numpy()
        sample[sample < 0.5] = 0
        sample[sample >= 0.5] = 1
        sample = sample.astype(np.uint8)
        sample = sample[:16]
        output_dir = f'{log_dir}/midi_sample_iter_{iteration}_small'
        single_channel_pianoroll_save_to_midi(sample, output_dir)


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
