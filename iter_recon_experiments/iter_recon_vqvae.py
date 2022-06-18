import os
import torch
import argparse
from utils.train_utils import set_seed, str2bool
from utils.data_utils import get_init_data
from torchvision import transforms

set_seed(1234)

from models.vqvae_utils import get_model_assets, train_one_epoch, \
    save_sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_act', type=str, default='sigmoid')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--K', type=int, default=512)
    parser.add_argument('--ex_spatial', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--ex_lifetime', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--qx_spatial', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--qx_lifetime', type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    log_dir = f'./logs/vqvae_logs/out_act_{args.out_act}_dim_{args.dim}_K_{args.K}'
    if args.ex_spatial:
        log_dir += '_ex_spatial'
    if args.ex_lifetime:
        log_dir += '_ex_lifetime'
    if args.qx_spatial:
        log_dir += '_qx_spatial'
    if args.qx_lifetime:
        log_dir += '_qx_lifetime'
    os.makedirs(log_dir, exist_ok=True)

    batch_size = 128

    model, optimizer = get_model_assets(dim=args.dim,
                                        K=args.K,
                                        image_size=32,
                                        out_act=args.out_act,
                                        ex_spatial=args.ex_spatial,
                                        ex_lifetime=args.ex_lifetime,
                                        qx_spatial=args.qx_spatial,
                                        qx_lifetime=args.qx_lifetime)

    if args.out_act == 'sigmoid':
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
        ])
    elif args.out_act == 'tanh':
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
            transforms.Normalize((0.5), (0.5)),
        ])

    train_data, _ = get_init_data(transform=transforms, dataset_name='mnist', batch_size=batch_size)

    total_epoch = 30
    for epoch in range(total_epoch):
        model.train()
        model, optimizer, avg_loss_recon, avg_loss_vq, avg_perplexity = train_one_epoch(model, optimizer, train_data)
        print(f'Epoch: {epoch}, Recon Loss: {avg_loss_recon:.4f}, '
              f'VQ Loss: {avg_loss_vq:.4f}, Perplexity: {avg_perplexity:.4f}')
        save_sample((model, optimizer), log_dir, epoch)
