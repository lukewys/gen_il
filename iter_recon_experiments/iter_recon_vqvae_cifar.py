import os
import torch
import argparse
from utils.train_utils import set_seed, str2bool
from utils.data_utils import get_init_data
from torchvision import transforms

set_seed(1234)

from models.vqvae_utils import get_model_assets, train_one_epoch, \
    DIFF_THRES, MAX_RECON_ITER, recon_fn, recon_till_converge, save_image, get_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_sample(model_assets, log_dir, iteration, thres=DIFF_THRES, max_iteration=MAX_RECON_ITER):
    model, optimizer = model_assets
    model.eval()
    sample_z = model.sample_z.to(device)
    image_size = model.image_size
    ch = model.ch
    sample_num = sample_z.shape[0]
    sample = recon_till_converge(model, recon_fn, sample_z,
                                 thres=thres, max_iteration=max_iteration, renorm='none').cpu()
    save_image(sample.view(sample_num, ch, image_size, image_size), f'{log_dir}/sample_iter_{iteration}_full' + '.png',
               nrow=32)
    save_image(sample.view(sample_num, ch, image_size, image_size)[:64],
               f'{log_dir}/sample_iter_{iteration}_small' + '.png', nrow=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--K', type=int, default=512)
    parser.add_argument('--ex_spatial', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--ex_lifetime', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--qx_spatial', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--qx_lifetime', type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    log_dir = f'./logs/vqvae_logs/cifar10_dim_{args.dim}_K_{args.K}'
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

    model, optimizer = get_model_assets(input_dim=3,
                                        dim=args.dim,
                                        K=args.K,
                                        image_size=32,
                                        out_act='tanh',
                                        ex_spatial=args.ex_spatial,
                                        ex_lifetime=args.ex_lifetime,
                                        qx_spatial=args.qx_spatial,
                                        qx_lifetime=args.qx_lifetime)

    train_data, _ = get_init_data(transform=get_transform('cifar10'), dataset_name='cifar10', batch_size=batch_size)

    total_epoch = 100
    for epoch in range(total_epoch):
        model.train()
        model, optimizer, avg_loss_recon, avg_loss_vq, avg_perplexity = train_one_epoch(model, optimizer, train_data)
        print(f'Epoch: {epoch}, Recon Loss: {avg_loss_recon:.4f}, '
              f'VQ Loss: {avg_loss_vq:.4f}, Perplexity: {avg_perplexity:.4f}')
        save_sample((model, optimizer), log_dir, epoch)
