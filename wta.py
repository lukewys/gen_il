import os
import torch
from wta_utils import get_model_assets, get_train_data, recon_till_converge, save_image, get_kernel_visualization, \
    SAMPLE_NUM, SAMPLE_Z

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 100


def train(model_assets, train_data):
    model, optimizer = model_assets
    model.train()
    total_epoch = 200
    for epoch in range(total_epoch):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_data):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch = model(data)
            loss = .5 * ((recon_batch - data) ** 2).sum() / BATCH_SIZE  # loss_function(recon_batch, data)
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
                   f'{log_dir}/sample_iter_{iteration}_thres_{str(1000)}full' + '.png', nrow=32)
    kernel_img = get_kernel_visualization(model)
    save_image(kernel_img.view(1, 1, 28, 28), f'{log_dir}/kernel_iter_{iteration}' + '.png', nrow=8)


if __name__ == '__main__':
    log_dir = './wta_logs'
    os.makedirs(log_dir, exist_ok=True)
    model_assets = get_model_assets()
    train_data = get_train_data()
    train(model_assets, train_data)

