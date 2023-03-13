import torch
import torch.utils.data
from torch import nn, optim
from torchvision import transforms
import numpy as np
import copy

from models.wta_utils import TOTAL_EPOCH
from .latent_wta_mse_image_utils import LatentWTA, get_data_config, evaluate, save_sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
