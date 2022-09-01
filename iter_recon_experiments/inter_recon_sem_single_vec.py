import os
import torch
import argparse
from utils.train_utils import set_seed
from utils.data_utils import get_init_data
from models.wta_utils import save_sample, train_one_epoch, evaluate
from models.sem_single_vec_utils import get_new_model, get_data_config

set_seed(1234)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--message_size', type=int, default=100)
    parser.add_argument('--voc_size', type=int, default=8)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--proj_hidden_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--hid_channels', type=int, default=32)
    parser.add_argument('--total_epoch', type=int, default=50)

    args = parser.parse_args()

    log_dir = f'./logs/sem_single_vec_logs/dataset_{args.dataset}_batch_size_{args.batch_size}_' \
              f'message_size_{args.message_size}_voc_size_{args.voc_size}_tau_{args.tau}_' \
              f'proj_hidden_dim_{args.proj_hidden_dim}_latent_dim_{args.latent_dim}_hid_channels_{args.hid_channels}_' \
              f'total_epoch_{args.total_epoch}'
    os.makedirs(log_dir, exist_ok=True)

    batch_size = args.batch_size

    data_config = get_data_config(args.dataset)

    train_data, test_data = get_init_data(transform=data_config['transform'], dataset_name=args.dataset,
                                          batch_size=batch_size, data_dir=args.data_dir)

    model, optimizer = get_new_model(image_size=data_config['image_size'], ch=data_config['ch'],
                                     out_act=data_config['out_act'], latent_dim=args.latent_dim,
                                     message_size=args.message_size, voc_size=args.voc_size, tau=args.tau,
                                     proj_hidden_dim=args.proj_hidden_dim)

    total_epoch = args.total_epoch
    for epoch in range(total_epoch):
        model.train()
        model, optimizer, avg_loss = train_one_epoch(model, optimizer, train_data)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, avg_loss))
        evaluate((model, optimizer), test_data, data_config['transform'], log_dir=log_dir, iteration=epoch)
        save_sample((model, optimizer), log_dir, epoch, data_config['transform'], save_kernel=False)
