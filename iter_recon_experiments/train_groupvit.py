import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils.train_utils import str2bool, set_seed
from utils.data_utils import get_init_data
import utils
from torchvision.utils import save_image
from models.group_autoencoder import GroupAutoencoder, build_scheduler, build_optimizer, ViTAutoencoder, \
    ViTCNNAutoencoder, GroupCNNAutoencoder

set_seed(1234)

from models.wta_utils import get_data_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize a 8 type color map
color_map_8 = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
    [0.5, 0.5, 0.5],
    [0.5, 0, 0.5]
]

color_map_16 = [
    [48 / 255, 96 / 255, 130 / 255],
    [91 / 255, 110 / 255, 225 / 255],
    [99 / 255, 155 / 255, 255 / 255],
    [95 / 255, 205 / 255, 228 / 255],
    [203 / 255, 219 / 255, 252 / 255],
    [255 / 255, 255 / 255, 255 / 255],
    [155 / 255, 173 / 255, 183 / 255],
    [132 / 255, 126 / 255, 135 / 255],
    [105 / 255, 106 / 255, 106 / 255],
    [89 / 255, 86 / 255, 82 / 255],
    [118 / 255, 66 / 255, 138 / 255],
    [172 / 255, 50 / 255, 50 / 255],
    [217 / 255, 87 / 255, 99 / 255],
    [215 / 255, 123 / 255, 186 / 255],
    [143 / 255, 151 / 255, 74 / 255],
    [138 / 255, 111 / 255, 48 / 255]
]

color_map_32 = [
    [1, 1, 1],
    [34 / 255, 32 / 255, 52 / 255],
    [69 / 255, 40 / 255, 60 / 255],
    [102 / 255, 57 / 255, 49 / 255],
    [143 / 255, 86 / 255, 59 / 255],
    [223 / 255, 113 / 255, 38 / 255],
    [217 / 255, 160 / 255, 102 / 255],
    [235 / 255, 195 / 255, 154 / 255],
    [251 / 255, 242 / 255, 54 / 255],
    [153 / 255, 229 / 255, 80 / 255],
    [106 / 255, 190 / 255, 48 / 255],
    [55 / 255, 148 / 255, 110 / 255],
    [75 / 255, 105 / 255, 47 / 255],
    [82 / 255, 75 / 255, 36 / 255],
    [50 / 255, 60 / 255, 57 / 255],
    [63 / 255, 63 / 255, 116 / 255],
    [48 / 255, 96 / 255, 130 / 255],
    [91 / 255, 110 / 255, 225 / 255],
    [99 / 255, 155 / 255, 255 / 255],
    [95 / 255, 205 / 255, 228 / 255],
    [203 / 255, 219 / 255, 252 / 255],
    [255 / 255, 255 / 255, 255 / 255],
    [155 / 255, 173 / 255, 183 / 255],
    [132 / 255, 126 / 255, 135 / 255],
    [105 / 255, 106 / 255, 106 / 255],
    [89 / 255, 86 / 255, 82 / 255],
    [118 / 255, 66 / 255, 138 / 255],
    [172 / 255, 50 / 255, 50 / 255],
    [217 / 255, 87 / 255, 99 / 255],
    [215 / 255, 123 / 255, 186 / 255],
    [143 / 255, 151 / 255, 74 / 255],
    [138 / 255, 111 / 255, 48 / 255]
]

color_map = [torch.tensor(color_map_16).float().to(device)]


def train_one_epoch(model, optimizer, train_data):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_data):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = F.mse_loss(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data.dataset),
                       100. * batch_idx / len(train_data), train_loss / (batch_idx + 1)))
    return model, optimizer, train_loss / len(train_data)


def evaluate(model_assets, test_data, transform, **kwargs):
    model, optimizer = model_assets
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_data):
            data = data.to(device)
            recon_batch = model(data)
            loss = F.mse_loss(recon_batch, data)
            test_loss += loss.item()

        print('====> Average test loss: {:.4f}'.format(
            test_loss / len(test_data)))

        data, _ = next(iter(test_data))
        data = data.to(device)
        recon = model(data)
        recon = recon.cpu().detach()
        image_size = model.image_size
        ch = data.shape[1]
        log_dir = kwargs['log_dir']
        iteration = kwargs['iteration']
        recon = utils.data_utils.denormalize(recon, transform)
        data = utils.data_utils.denormalize(data, transform)
        save_image(recon.view(recon.shape[0], ch, image_size, image_size)[:64],
                   f'{log_dir}/recon_iter_{iteration}' + '.png', nrow=8)
        save_image(data.view(data.shape[0], ch, image_size, image_size)[:64],
                   f'{log_dir}/test_gt_iter_{iteration}' + '.png', nrow=8)

        if isinstance(model, GroupAutoencoder) or isinstance(model, GroupCNNAutoencoder):
            for i in range(1):
                attn_map_colored = model.get_colored_attn_map(data, color_map[i], layer=i)
                attn_map_colored = F.interpolate(attn_map_colored, size=(image_size, image_size),
                                                 mode='bilinear', align_corners=False)
                save_image(attn_map_colored.view(attn_map_colored.shape[0], ch, image_size, image_size)[:64],
                           f'{log_dir}/attn_map_{i}_iter_{iteration}' + '.png', nrow=8)
                blended_img = attn_map_colored * 0.5 + data * 0.5
                save_image(blended_img.view(blended_img.shape[0], ch, image_size, image_size)[:64],
                           f'{log_dir}/blended_img_{i}_iter_{iteration}' + '.png', nrow=8)

    return model, optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--data_dir', type=str, default='./data')

    args = parser.parse_args()

    log_dir = f'./logs/groupAE/dataset_{args.dataset}_batch_size_{args.batch_size}_GroupCNNAutoencoder_16_tokens_1layer_hard_assign'
    os.makedirs(log_dir, exist_ok=True)

    batch_size = args.batch_size

    data_config = get_data_config(args.dataset)

    train_data, test_data = get_init_data(transform=data_config['transform'], dataset_name=args.dataset,
                                          batch_size=batch_size, data_dir=args.data_dir)

    model = GroupCNNAutoencoder(image_size=data_config['image_size'],
                                patch_size=4,
                                in_chans=data_config['ch'], ).to(device)

    total_epoch = 50
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(total_epoch, optimizer, n_iter_per_epoch=len(train_data))

    print(len(train_data))

    evaluate((model, optimizer), test_data, data_config['transform'], log_dir=log_dir, iteration=-1)

    for epoch in range(total_epoch):
        model.train()
        model, optimizer, avg_loss = train_one_epoch(model, optimizer, train_data)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, avg_loss))
        evaluate((model, optimizer), test_data, data_config['transform'], log_dir=log_dir, iteration=epoch)
