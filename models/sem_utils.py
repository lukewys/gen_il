import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical
import torch.nn.functional as F
import torch.optim as optim
import copy
from .wta_utils import WTA, spatial_sparsity, lifetime_sparsity, train, train_one_epoch, \
    train_with_teacher, gen_data, get_train_data_next_iter, get_linear_probe_model, BATCH_SIZE, \
    get_transform, DIFF_THRES, recon_till_converge, recon_fn, MAX_RECON_ITER, save_image, get_kernel_visualization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://github.com/facebookresearch/EGG/blob/3a429c27b798a24b12b05486f8832d4a82ea3327/egg/core/gs_wrappers.py#L43
def gumbel_softmax_sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        training: bool = True,
        straight_through: bool = False,
):
    size = logits.size()
    if not training:
        indexes = logits.argmax(dim=-1)
        one_hot = torch.zeros_like(logits).reshape(-1, size[-1])
        one_hot.scatter_(1, indexes.reshape(-1, 1), 1)
        one_hot = one_hot.reshape(*size)
        return one_hot

    sample = RelaxedOneHotCategorical(logits=logits, temperature=temperature).rsample()

    if straight_through:
        size = sample.size()
        indexes = sample.argmax(dim=-1)
        hard_sample = torch.zeros_like(sample).reshape(-1, size[-1])
        hard_sample.scatter_(1, indexes.reshape(-1, 1), 1)
        hard_sample = hard_sample.reshape(*size)

        sample = sample + (hard_sample - sample).detach()
    return sample


class GumbelSoftmaxBottleneck(nn.Module):
    def __init__(
            self,
            temperature: float = 1.0,
            trainable_temperature: bool = False,
            straight_through: bool = False,
            decay: float = 0.9,
            minimum: float = 0.1,
            num_decompose: int = 1,  # decompose the message into multiple onehot
    ):
        super(GumbelSoftmaxBottleneck, self).__init__()
        self.straight_through = straight_through

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

        self.decay = decay
        self.minimum = minimum
        self.num_decompose = num_decompose

    def update_temperature(self):
        self.temperature = max(
            self.minimum, self.temperature * self.decay
        )

    def forward(self, logits: torch.Tensor):
        if self.num_decompose > 1:
            original_shape = logits.shape
            decomposed_shape = logits.shape[:-1] + (self.num_decompose, -1)
            logits = logits.reshape(decomposed_shape)
            discrete_logits = gumbel_softmax_sample(
                logits, self.temperature, self.training, self.straight_through
            )
            discrete_logits = discrete_logits.reshape(original_shape)
            return discrete_logits

        else:
            return gumbel_softmax_sample(
                logits, self.temperature, self.training, self.straight_through
            )


class SoftmaxBottleneck(nn.Module):
    def __init__(
            self,
            temperature: float = 1.0,
    ):
        super(SoftmaxBottleneck, self).__init__()
        self.temperature = temperature

    def forward(self, z):
        ori_shape = z.shape
        z = z.reshape(z.shape[0], z.shape[1], -1)
        z = F.softmax(z / self.temperature, -1).reshape(ori_shape)
        return z


class SEM(WTA):
    def __init__(self, sz=64, code_sz=128,
                 lifetime_sparsity_rate=0.05,
                 channel_sparsity_rate=1, ch=1,
                 image_size=28, sample_num=1024,
                 temp=0.5, z_spatial_sparsity=False,
                 z_lifetime_sparsity=False,
                 gumbel_softmax=False,
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

        if gumbel_softmax:
            self.sem = GumbelSoftmaxBottleneck(
                temperature=temp,
                straight_through=True,
                num_decompose=1,
            )
        else:
            self.sem = SoftmaxBottleneck(
                temperature=temp,
            )

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
