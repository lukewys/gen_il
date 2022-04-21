# prerequisites
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import copy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 500

# loss
criterion = nn.BCELoss()
z_dim = 100
mnist_dim = 28 * 28
lr = 0.0002
NUM_WORKERS = 8
SAMPLE_NUM = 1024
SAMPLE_Z = torch.randn(SAMPLE_NUM, z_dim, 1, 1).to(device)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


def D_train(x, G, D, D_optimizer):
    # =======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, 1, 28, 28), torch.ones(x.shape[0])
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = Variable(torch.randn(x.shape[0], z_dim, 1, 1).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(x.shape[0]).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(x, G, D, G_optimizer):
    # =======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(x.shape[0], z_dim, 1, 1).to(device))
    y = Variable(torch.ones(x.shape[0]).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


def train(model_assets, train_data, train_extend):
    G, D, G_optimizer, D_optimizer = model_assets
    G.train()
    D.train()
    total_epoch = 100
    total_epoch = int(np.round(train_extend * total_epoch))
    for epoch in range(total_epoch):
        D_losses, G_losses = [], []
        for batch_idx, (x, _) in enumerate(train_data):
            D_losses.append(D_train(x, G, D, D_optimizer))
            G_losses.append(G_train(x, G, D, G_optimizer))
        if epoch % 10 == 0:
            print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                epoch, total_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

    return G, D, G_optimizer, D_optimizer


def train_with_teacher(new_model_assets, old_model_assets, steps):
    G, D, G_optimizer, D_optimizer = new_model_assets
    G.train()
    D.train()
    D_losses, G_losses = [], []

    for batch_idx in range(steps):
        x = gen_data(old_model_assets, BATCH_SIZE, 1).to(device)
        D_losses.append(D_train(x, G, D, D_optimizer))
        G_losses.append(G_train(x, G, D, G_optimizer))
        if batch_idx % 100 == 0:
            print('steps [%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                batch_idx, steps, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

    return G, D, G_optimizer, D_optimizer


def get_train_data(batch_size=BATCH_SIZE):
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=NUM_WORKERS)

    return train_loader


def get_new_model():
    # build network
    G = Generator().to(device)
    G.apply(weights_init)
    D = Discriminator().to(device)
    D.apply(weights_init)

    # optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    return G, D, G_optimizer, D_optimizer


FIX_MODEL_INIT = get_new_model()


def get_model_assets(model_assets=None, reset_model=True, use_same_init=True):
    if reset_model:
        if use_same_init:
            return copy.deepcopy(FIX_MODEL_INIT)
        else:
            return get_new_model()
    else:
        return model_assets


def get_train_data_next_iter(train_data, data_generated, add_old_dataset=False, batch_size=BATCH_SIZE):
    if add_old_dataset:
        old_data_all = []
        for batch_idx, (data, _) in enumerate(train_data):
            old_data_all.append(data)
        old_data_all = torch.cat(old_data_all, dim=0)
        data_combined = torch.cat([old_data_all.view(-1, 1, 28, 28), data_generated], dim=0)
        train_loader_new = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_combined, data_combined),
            batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        return train_loader_new
    else:
        train_loader_new = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_generated, data_generated),
            batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader_new


def gen_data(model_assets, gen_batch_size, gen_num_batch):
    data_all = []
    G, D, G_optimizer, D_optimizer = model_assets
    G.eval()
    with torch.no_grad():
        for _ in range(gen_num_batch):
            test_z = torch.randn(gen_batch_size, z_dim, 1, 1).to(device)
            sample = G(test_z).cpu()
            data_all.append(sample)
    data_all = torch.cat(data_all, dim=0)
    return data_all


def save_sample(model_assets, log_dir, iteration):
    G, D, G_optimizer, D_optimizer = model_assets
    with torch.no_grad():
        sample = G(SAMPLE_Z).cpu()
    sample += 1
    sample *= 0.5
    save_image(sample.view(SAMPLE_NUM, 1, 28, 28), f'{log_dir}/sample_iter_{iteration}_full' + '.png', nrow=32)
    save_image(sample.view(SAMPLE_NUM, 1, 28, 28)[:64], f'{log_dir}/sample_iter_{iteration}_small' + '.png', nrow=8)
