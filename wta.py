import os
import torch
import argparse
from utils.train_utils import set_seed
from utils.data_utils import get_init_data

set_seed(1234)

from models.wta_utils import get_model_assets, recon_till_converge, save_image, get_kernel_visualization, \
    SAMPLE_NUM, SAMPLE_Z, get_transform


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model_assets, train_data):
    model, optimizer = model_assets
    model.train()
    total_epoch = 50
    for epoch in range(total_epoch):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_data):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch = model(data)
            loss = .5 * ((recon_batch - data) ** 2).sum() / batch_size  # loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_data)))
        save_sample(model_assets, log_dir, epoch)


def save_sample(model_assets, log_dir, iteration):
    model, optimizer = model_assets
    thresholds = [1e-1, 1e-2, 1e-3, 1e-5, 1e-9]
    for thres in thresholds:
        sample = recon_till_converge(model, SAMPLE_Z, thres=thres, max_iteration=1000).cpu()
        save_image(sample.view(SAMPLE_NUM, 1, 28, 28),
                   f'{log_dir}/sample_iter_{iteration}_thres_{str(thres)}full' + '.png', nrow=32)
    kernel_img = get_kernel_visualization(model)
    save_image(kernel_img.view(model.code_sz, 1, 28, 28), f'{log_dir}/kernel_iter_{iteration}' + '.png', nrow=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lifetime_sparsity_rate', type=float, default=0.05)
    parser.add_argument('--channel_sparsity_rate', type=float, default=1.0)
    parser.add_argument('--code_sz', type=int, default=128)
    parser.add_argument('--sz', type=int, default=64)
    args = parser.parse_args()

    log_dir = f'./wta_logs/batch_size_{args.batch_size}_lifetime_sparsity_rate_{args.lifetime_sparsity_rate}_' \
              f'channel_sparsity_rate_{args.channel_sparsity_rate}_code_sz_{args.code_sz}_sz_{args.sz}'
    os.makedirs(log_dir, exist_ok=True)

    batch_size = args.batch_size

    model_assets = get_model_assets()
    train_data, _ = get_init_data(transform=get_transform('mnist'), dataset_name='mnist', batch_size=batch_size)
    train(model_assets, train_data)
