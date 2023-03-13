import os
import torch
import argparse
from utils.train_utils import str2bool, set_seed
from utils.data_utils import get_init_data

set_seed(1234)

from models.sparse_gan_utils import get_model_assets, train, get_data_config, save_sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=1024)

    args = parser.parse_args()

    log_dir = f'./logs/sparse_gan_standalone_training/dataset_{args.dataset}_64_sparsity_at_16x16'
    os.makedirs(log_dir, exist_ok=True)

    batch_size = args.batch_size

    data_config = get_data_config(args.dataset)

    train_data, test_data = get_init_data(transform=data_config['transform'], dataset_name=args.dataset,
                                          batch_size=batch_size, data_dir=args.data_dir)

    model_assets = get_model_assets(nc=data_config['ch'])

    model_assets = train(model_assets, train_data, train_extend=1, log_dir=log_dir, transform=data_config['transform'])

    save_sample(model_assets, log_dir, 0, data_config['transform'], dataset_name=args.dataset)
