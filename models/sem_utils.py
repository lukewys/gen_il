import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from .wta_utils import weights_init, WTA, spatial_sparsity, lifetime_sparsity, train, train_one_epoch, \
    train_with_teacher, gen_data, get_train_data_next_iter, get_linear_probe_model, BATCH_SIZE, \
    get_transform, DIFF_THRES, recon_till_converge, recon_fn, MAX_RECON_ITER, save_image, get_kernel_visualization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SEM(WTA):
    def __init__(self, sz=64, code_sz=128,
                 lifetime_sparsity_rate=0.05,
                 channel_sparsity_rate=1, ch=1,
                 image_size=28, sample_num=1024,
                 temp=0.5, z_spatial_sparsity=False,
                 z_lifetime_sparsity=False,
                 **kwargs):
        super(SEM, self).__init__(sz, code_sz,
                                  lifetime_sparsity_rate,
                                  channel_sparsity_rate, ch,
                                  image_size, sample_num,
                                  out_act='none',  # SEM only works with no activation
                                  **kwargs)
        self.temp = temp
        self.z_spatial_sparsity = z_spatial_sparsity
        self.z_lifetime_sparsity = z_lifetime_sparsity

    def sem(self, z):
        ori_shape = z.shape
        z = z.reshape(z.shape[0], z.shape[1], -1)
        z = F.softmax(z / self.temp, -1).reshape(ori_shape)
        return z

    def forward(self, x, **kwargs):
        z = self.encode(x)
        if self.z_spatial_sparsity:
            z = spatial_sparsity(z)
        if self.z_lifetime_sparsity and self.training:
            z = lifetime_sparsity(z)
        z = self.sem(z)
        out = self.decode(z)
        return out


def get_new_model(**kwargs):
    model = SEM(**kwargs).to(device)
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


def save_sample(model_assets, log_dir, iteration, thres=DIFF_THRES, max_iteration=MAX_RECON_ITER):
    model, optimizer = model_assets
    model.eval()
    sample_z = model.sample_z.to(device)
    image_size = model.image_size
    ch = model.ch
    sample_num = sample_z.shape[0]
    sample = recon_till_converge(model, recon_fn, sample_z, thres=thres, max_iteration=max_iteration,
                                 renorm='0_to_1').cpu()
    save_image(sample.view(sample_num, ch, image_size, image_size), f'{log_dir}/sample_iter_{iteration}_full' + '.png',
               nrow=32)
    save_image(sample.view(sample_num, ch, image_size, image_size)[:64],
               f'{log_dir}/sample_iter_{iteration}_small' + '.png', nrow=8)
    kernel_img = get_kernel_visualization(model)
    save_image(kernel_img.view(model.code_sz, ch, image_size, image_size),
               f'{log_dir}/kernel_iter_{iteration}' + '.png', nrow=8)
