# https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/vqvae.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torch.autograd import Function
import utils.data_utils
import copy
from utils.iter_recon_utils import recon_till_converge
from .wta_utils import spatial_sparsity, lifetime_sparsity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
NUM_WORKERS = 8
TOTAL_EPOCH = 30
DIFF_THRES = 1e-6
MAX_RECON_ITER = 100


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        # b,h,w,d
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                                    inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
                                           index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                   .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply


def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)

    def forward(self, z_e_x):
        # b,d,h,w
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()  # b,h,w,d
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar, indices


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


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim=1, dim=256, K=512, out_act='sigmoid',
                 image_size=32, sample_num=1024,
                 ex_spatial=False, ex_lifetime=False, qx_spatial=False, qx_lifetime=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
        )

        if out_act == 'sigmoid':
            self.decoder = nn.Sequential(self.decoder, nn.Sigmoid())
        elif out_act == 'tanh':
            self.decoder = nn.Sequential(self.decoder, nn.Tanh())
        elif out_act == 'none':
            pass
        else:
            raise ValueError('Unknown output activation function: {}'.format(out_act))
        self.out_act = out_act
        self.K = K
        self.ch = input_dim
        self.image_size = image_size

        self.ex_spatial = ex_spatial
        self.ex_lifetime = ex_lifetime
        self.qx_spatial = qx_spatial
        self.qx_lifetime = qx_lifetime

        self.sample_z = torch.rand(sample_num, input_dim, image_size, image_size)
        if out_act == 'tanh':
            self.sample_z = self.sample_z * 2 - 1

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        if self.ex_spatial:
            z_e_x = spatial_sparsity(z_e_x)
        if self.ex_lifetime and self.training:
            z_e_x = lifetime_sparsity(z_e_x)
        z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x)
        if self.qx_spatial:
            z_q_x_st = spatial_sparsity(z_q_x_st)
        if self.qx_lifetime and self.training:
            z_q_x_st = lifetime_sparsity(z_q_x_st)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, indices


def train_one_epoch(model, optimizer, train_data):
    beta = 1

    total_loss_recon = 0
    total_loss_vq = 0
    total_perplexity = 0
    for batch_idx, (data, _) in enumerate(train_data):
        images = data.to(device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x, indices = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + beta * loss_commit
        loss.backward()

        indices = indices.reshape(-1).detach().cpu().numpy()
        indices = np.array(np.eye(model.K)[indices])
        avg_probs = np.mean(indices, axis=0)
        perplexity = np.exp(- np.sum(avg_probs * np.log(avg_probs + 1e-10)))

        total_loss_recon += loss_recons.item()
        total_loss_vq += loss_vq.item()
        total_perplexity += perplexity

        optimizer.step()

    avg_loss_recon = total_loss_recon / len(train_data)
    avg_loss_vq = total_loss_vq / len(train_data)
    avg_perplexity = total_perplexity / len(train_data)
    return model, optimizer, avg_loss_recon, avg_loss_vq, avg_perplexity


def train(model_assets, train_data, train_extend):
    model, optimizer = model_assets
    model.train()

    total_epoch = int(np.round(train_extend * TOTAL_EPOCH))

    for epoch in range(total_epoch):
        model, optimizer, avg_loss_recon, avg_loss_vq, avg_perplexity = train_one_epoch(model, optimizer, train_data)
        print(f'Epoch: {epoch}, Recon Loss: {avg_loss_recon:.4f}, '
              f'VQ Loss: {avg_loss_vq:.4f}, Perplexity: {avg_perplexity:.4f}')
    return model, optimizer


def evaluate(model_assets, test_data):
    model, optimizer = model_assets
    model.eval()
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in test_data:
            images = images.to(device)
            x_tilde, z_e_x, z_q_x, indices = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(test_data)
        loss_vq /= len(test_data)

    print(f'Eval Recon Loss: {loss_recons.item():.4f}, VQ Loss: {loss_vq.item():.4f}')
    return model, optimizer


def train_with_teacher(new_model_assets, old_model_assets, steps, **kwargs):
    model, optimizer = new_model_assets
    model.train()
    beta = 1

    total_loss_recon = 0
    total_loss_vq = 0
    total_perplexity = 0
    for batch_idx in range(steps):
        images = gen_data(old_model_assets, BATCH_SIZE, 1).to(device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x, indices = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + beta * loss_commit
        loss.backward()

        indices = indices.reshape(-1).detach().cpu().numpy()
        indices = np.array(np.eye(model.K)[indices])
        avg_probs = np.mean(indices, axis=0)
        perplexity = np.exp(- np.sum(avg_probs * np.log(avg_probs + 1e-10)))

        total_loss_recon += loss_recons.item()
        total_loss_vq += loss_vq.item()
        total_perplexity += perplexity

        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Step: {batch_idx}, Recon Loss: {(total_loss_recon / batch_idx):.4f}, '
                  f'VQ Loss: {(total_loss_vq / batch_idx):.4f}, '
                  f'Perplexity: {(total_perplexity / batch_idx):.4f})')

    return model, optimizer


def get_transform(dataset_name):
    if dataset_name in ['mnist', 'fashion-mnist', 'kuzushiji']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
            transforms.Normalize((0.5), (0.5)),
        ])
    elif dataset_name == 'omniglot':
        return transforms.Compose([
            transforms.ToTensor(),
            utils.data_utils.flip_image_value,
            transforms.Resize(32),
            transforms.Normalize((0.5), (0.5)),
        ])
    elif dataset_name in ['cifar10', 'cifar100', 'wikiart']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))


FIX_MODEL_INIT = None


def get_new_model(**kwargs):
    model = VectorQuantizedVAE(**kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    return model, optimizer


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


def recon_fn(model, images):
    x_tilde, _, _, _ = model(images)
    return x_tilde


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
