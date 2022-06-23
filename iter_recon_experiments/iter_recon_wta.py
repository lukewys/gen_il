import os
import torch
import argparse
from torchvision import transforms
from utils.train_utils import set_seed
from utils.data_utils import get_init_data

set_seed(1234)

from models.wta_utils import get_model_assets, save_sample, train_one_epoch, get_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lifetime_sparsity_rate', type=float, default=0.05)
    parser.add_argument('--channel_sparsity_rate', type=float, default=1.0)
    parser.add_argument('--code_sz', type=int, default=128)
    parser.add_argument('--sz', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='mnist')  # TODO: unused
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--net_type', type=str, default='wta')

    args = parser.parse_args()

    log_dir = f'./logs/wta_logs/batch_size_{args.batch_size}_lifetime_sparsity_rate_{args.lifetime_sparsity_rate}_' \
              f'channel_sparsity_rate_{args.channel_sparsity_rate}_code_sz_{args.code_sz}_sz_{args.sz}' \
              f'_image_size_{args.image_size}_kernel_size_{args.kernel_size}_net_type_{args.net_type}'
    os.makedirs(log_dir, exist_ok=True)

    batch_size = args.batch_size

    model_assets = get_model_assets(lifetime_sparsity_rate=args.lifetime_sparsity_rate,
                                    channel_sparsity_rate=args.channel_sparsity_rate,
                                    code_sz=args.code_sz,
                                    sz=args.sz,
                                    image_size=args.image_size,
                                    net_type=args.net_type,
                                    kernel_size=args.kernel_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.image_size),
    ])

    train_data, _ = get_init_data(transform=transform, dataset_name='mnist', batch_size=batch_size)

    total_epoch = 50
    for epoch in range(total_epoch):
        model, optimizer = model_assets
        model.train()
        model, optimizer, avg_loss = train_one_epoch(model, optimizer, train_data)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, avg_loss))
        save_sample((model, optimizer), log_dir, epoch, save_kernel=args.net_type == 'wta')
