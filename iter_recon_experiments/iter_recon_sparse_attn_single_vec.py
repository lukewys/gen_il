import os
import torch
import argparse
import sys

sys.path.append('/data/creativity_generation/generative_IL')

from utils.train_utils import set_seed
from utils.data_utils import get_init_data
from models.wta_utils import save_sample, train_one_epoch, evaluate
from models.attn_single_vec_utils import get_new_model, get_data_config

set_seed(1234)
from easydict import EasyDict

if __name__ == '__main__':

    args = EasyDict()
    args.batch_size = 2048
    args.num_topk = 16
    args.dataset = 'mnist'
    args.data_dir = './data'
    args.voc_size = 128
    args.proj_hidden_dim = 128
    args.latent_dim = 64
    args.hid_channels = 32
    args.total_epoch = 40

    log_dir = f'./logs/attn_single_vec_logs/dataset_{args.dataset}_batch_size_{args.batch_size}_' \
              f'voc_size_{args.voc_size}_num_topk_{args.num_topk}' \
              f'proj_hidden_dim_{args.proj_hidden_dim}_latent_dim_{args.latent_dim}_hid_channels_{args.hid_channels}_' \
              f'total_epoch_{args.total_epoch}'
    os.makedirs(log_dir, exist_ok=True)

    batch_size = args.batch_size

    data_config = get_data_config(args.dataset)

    train_data, test_data = get_init_data(transform=data_config['transform'], dataset_name=args.dataset,
                                          batch_size=batch_size, data_dir=args.data_dir)

    model, optimizer = get_new_model(image_size=data_config['image_size'], ch=data_config['ch'],
                                     out_act=data_config['out_act'], latent_dim=args.latent_dim,
                                     voc_size=args.voc_size, num_topk=args.num_topk,
                                     proj_hidden_dim=args.proj_hidden_dim, hid_channels=args.hid_channels)

    total_epoch = args.total_epoch
    for epoch in range(total_epoch):
        model.train()
        model, optimizer, avg_loss = train_one_epoch(model, optimizer, train_data)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, avg_loss))
        evaluate((model, optimizer), test_data, data_config['transform'], log_dir=log_dir, iteration=epoch)
        save_sample((model, optimizer), log_dir, epoch, data_config['transform'], save_kernel=False,
                    renorm='none', no_renorm_last_iter=False, dataset_name='mnist')
