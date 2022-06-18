import os
import torch
import argparse
from utils.train_utils import set_seed, str2bool
from utils.data_utils import get_init_data

set_seed(1234)

from models.vae_utils import get_model_assets, train_one_epoch, get_transform, BATCH_SIZE
from models.wta_utils import DIFF_THRES, MAX_RECON_ITER, save_image, recon_till_converge

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_z = torch.rand(1024, 1, 32, 32).to(device)

def recon_fn(model, image):
    recon_batch, mu, logvar = model(image)
    return recon_batch


def save_sample(model, log_dir, iteration, thres=DIFF_THRES, max_iteration=MAX_RECON_ITER, renorm=False):
    model.eval()
    image_size = 32
    ch = 1
    sample_num = sample_z.shape[0]
    if renorm:
        sample = recon_till_converge(model, recon_fn, sample_z, thres=thres, max_iteration=max_iteration,
                                     renorm='0_to_1').cpu()
    else:
        sample = recon_till_converge(model, recon_fn, sample_z, thres=thres, max_iteration=max_iteration).cpu()
    save_image(sample.view(sample_num, ch, image_size, image_size), f'{log_dir}/sample_iter_{iteration}_full' + '.png',
               nrow=32)
    save_image(sample.view(sample_num, ch, image_size, image_size)[:64],
               f'{log_dir}/sample_iter_{iteration}_small' + '.png', nrow=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--renorm', type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    log_dir = f'./logs/vae_logs'
    if args.renorm:
        log_dir += '_renorm'
    os.makedirs(log_dir, exist_ok=True)

    model_assets = get_model_assets()
    train_data, _ = get_init_data(transform=get_transform('mnist'), dataset_name='mnist', batch_size=BATCH_SIZE)

    total_epoch = 100
    for epoch in range(total_epoch):
        model, optimizer = model_assets
        model.train()
        model, optimizer, avg_loss = train_one_epoch(model, optimizer, train_data)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, avg_loss))
        save_sample(model, log_dir, epoch, thres=DIFF_THRES, max_iteration=MAX_RECON_ITER, renorm=False)
