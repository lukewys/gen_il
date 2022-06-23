import os
import torch
import argparse
from utils.train_utils import set_seed, str2bool
from utils.data_utils import get_init_data

set_seed(1234)

from models.sem_utils import get_model_assets, save_sample, train_one_epoch, get_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--z_spatial_sparsity', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--z_lifetime_sparsity', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--gumbel_softmax', type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    log_dir = f'./logs/sem_logs/temp_{args.temp}'
    if args.z_spatial_sparsity:
        log_dir += '_z_spatial_sparsity'
    if args.z_lifetime_sparsity:
        log_dir += '_z_lifetime_sparsity'
    if args.gumbel_softmax:
        log_dir += '_gumbel_softmax'
    os.makedirs(log_dir, exist_ok=True)

    batch_size = 100

    model_assets = get_model_assets(temp=args.temp,
                                    z_spatial_sparsity=args.z_spatial_sparsity,
                                    z_lifetime_sparsity=args.z_lifetime_sparsity,
                                    gumbel_softmax=args.gumbel_softmax)
    train_data, _ = get_init_data(transform=get_transform('mnist'), dataset_name='mnist', batch_size=batch_size)

    total_epoch = 50
    for epoch in range(total_epoch):
        model, optimizer = model_assets
        model.train()
        model, optimizer, avg_loss = train_one_epoch(model, optimizer, train_data)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, avg_loss))
        save_sample((model, optimizer), log_dir, epoch)
