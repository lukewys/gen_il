import os
import torch
import argparse
from utils.train_utils import str2bool, set_seed
from utils.data_utils import get_init_data
from models.pretrain_vqvae import get_pretrained_vqvae

set_seed(1234)

from models.latent_wta_mse_image_utils import get_model_assets, save_sample, train_one_epoch, get_transform, get_data_config, evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lifetime_sparsity_rate', type=float, default=0.05)
    parser.add_argument('--channel_sparsity_rate', type=float, default=1.0)
    parser.add_argument('--code_sz', type=int, default=256)
    parser.add_argument('--sz', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--net_type', type=str, default='wta')
    parser.add_argument('--renorm', type=str, default='none')
    parser.add_argument('--no_renorm_last_iter', type=str2bool, nargs='?', const=True, default=False,
                        help='no_renorm_last_iter')
    parser.add_argument('--data_dir', type=str, default='./data')

    args = parser.parse_args()

    log_dir = f'./logs/latent_wta_mse_image_logs/dataset_{args.dataset}_batch_size_{args.batch_size}_' \
              f'lifetime_sparsity_rate_{args.lifetime_sparsity_rate}_' \
              f'channel_sparsity_rate_{args.channel_sparsity_rate}_code_sz_{args.code_sz}_sz_{args.sz}' \
              f'_net_type_{args.net_type}'
    os.makedirs(log_dir, exist_ok=True)

    batch_size = args.batch_size

    data_config = get_data_config(args.dataset)

    train_data, test_data = get_init_data(transform=data_config['transform'], dataset_name=args.dataset,
                                          batch_size=batch_size, data_dir=args.data_dir)

    if args.dataset == 'mnist':
        model_path = '/data/creativity_generation/generative_IL/pretrained_vqvae/pytorch-vqvae-master/models/models_mnist/best.pt'
        vqvae = get_pretrained_vqvae(1, model_path)
    elif args.dataset == 'cifar10':
        model_path = '/data/creativity_generation/generative_IL/pretrained_vqvae/pytorch-vqvae-master/models/models_cifar/best.pt'
        vqvae = get_pretrained_vqvae(3, model_path)

    model, optimizer = get_model_assets(lifetime_sparsity_rate=args.lifetime_sparsity_rate,
                                        channel_sparsity_rate=args.channel_sparsity_rate,
                                        code_sz=args.code_sz,
                                        sz=args.sz,
                                        image_size=data_config['image_size'],
                                        image_ch=data_config['ch'],
                                        out_act='none',
                                        net_type=args.net_type,
                                        vqvae=vqvae)

    total_epoch = 50



    for epoch in range(total_epoch):
        model.train()
        model, optimizer, avg_loss = train_one_epoch(model, optimizer, train_data)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, avg_loss))
        evaluate((model, optimizer), test_data, data_config['transform'], log_dir=log_dir, iteration=epoch)
        save_sample((model, optimizer), log_dir, epoch, data_config['transform'],
                    dataset_name=args.dataset,
                    renorm=args.renorm,
                    no_renorm_last_iter=False,
                    save_kernel=False)
