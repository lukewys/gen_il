from __future__ import print_function
import copy
import torch
import torch.utils.data
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

import utils.data_utils
from evaluate.linear_probe import LinearProbeModel

torch.manual_seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 500
NUM_WORKERS = 8
SAMPLE_NUM = 128
SAMPLE_Z = torch.randn(SAMPLE_NUM, 20).to(device)
TOTAL_EPOCH = 100


# https://github.com/YannDubs/disentangling-vae

def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {nn.LeakyReLU: "leaky_relu", nn.ReLU: "relu", nn.Tanh: "tanh",
              nn.Sigmoid: "sigmoid", nn.Softmax: "sigmoid"}
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def linear_init(layer, activation="relu"):
    """Initialize a linear layer.
    Args:
        layer (nn.Linear): parameters to initialize.
        activation (`torch.nn.modules.activation` or str, optional) activation that
            will be used on the `layer`.
    """
    x = layer.weight

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity='leaky_relu')
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity='relu')
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def weights_init(module):
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        # TO-DO: check litterature
        linear_init(module)
    elif isinstance(module, nn.Linear):
        linear_init(module)


class EncoderBurgess(nn.Module):
    def __init__(self, img_size=(1, 32, 32), latent_dim=10):
        r"""Encoder of the model proposed in [1].
        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_dim : int
            Dimensionality of latent output.
        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar


class DecoderBurgess(nn.Module):
    def __init__(self, img_size=(1, 32, 32), latent_dim=10):
        r"""Decoder of the model proposed in [1].
        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_dim : int
            Dimensionality of latent output.
        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        """
        Class which defines model and forward pass.
        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()
        img_size = (1, 32, 32)

        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError(
                "{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(
                    img_size))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = EncoderBurgess(img_size, self.latent_dim)
        self.decoder = DecoderBurgess(img_size, self.latent_dim)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps
        # if self.training:
        #     std = torch.exp(0.5 * logvar)
        #     eps = torch.randn_like(std)
        #     return mean + std * eps
        # else:
        #     # Reconstruction mode
        #     return mean

    def forward(self, x):
        """
        Forward pass of model.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x.view(-1, 1, 32, 32))
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        mu, logvar = latent_dist
        return reconstruct, mu, logvar

    def encode(self, x):
        return self.encoder(x.view(-1, 1, 32, 32))

    def decode(self, latent_sample):
        return self.decoder(latent_sample)

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train_one_epoch(model, optimizer, train_data):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_data):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return model, optimizer, train_loss / len(train_data.dataset)


def train(model_assets, train_data, train_extend):
    total_epoch = int(np.round(train_extend * TOTAL_EPOCH))
    model, optimizer = model_assets
    model.train()
    for epoch in range(total_epoch):
        model, optimizer, training_loss = train_one_epoch(model, optimizer, train_data)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, training_loss))
    return model, optimizer


def evaluate(model_assets, test_data, transform, **kwargs):
    return  # TODO: implement this


def train_with_teacher(new_model_assets, old_model_assets, steps, **kwargs):
    model, optimizer = new_model_assets
    model.train()
    train_loss = []
    for batch_idx in range(steps):
        data = gen_data(old_model_assets, BATCH_SIZE, 1).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        if batch_idx % 100 == 0:
            print('====> Step: {} Average loss: {:.4f}'.format(
                batch_idx, np.mean(train_loss) / BATCH_SIZE))

    return model, optimizer


def mnist_subset(dataset, classes_to_use):
    # classes_to_use: a list of class
    data = dataset.data.numpy()
    targets = dataset.targets.numpy()

    data_to_keep = []
    targets_to_keep = []
    for i in range(len(targets)):
        if targets[i] in classes_to_use:
            data_to_keep.append(data[i])
            targets_to_keep.append(targets[i])

    dataset.data = torch.tensor(np.stack(data_to_keep, axis=0))
    dataset.targets = torch.tensor(np.array(targets_to_keep))
    return dataset


def get_transform(dataset_name):
    if dataset_name in ['mnist', 'fashion-mnist', 'kuzushiji']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
        ])
    elif dataset_name == 'omniglot':
        return transforms.Compose([
            transforms.ToTensor(),
            utils.data_utils.flip_image_value,
            transforms.Resize(32),
        ])
    elif dataset_name in ['cifar10', 'cifar100', 'wikiart']:
        raise NotImplementedError  # TODO


def get_new_model():
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


FIX_MODEL_INIT = None


def get_model_assets(model_assets=None, reset_model=True, use_same_init=True):
    global FIX_MODEL_INIT
    if reset_model:
        if use_same_init and FIX_MODEL_INIT is None:
            FIX_MODEL_INIT = get_new_model()
        if use_same_init:
            return copy.deepcopy(FIX_MODEL_INIT)
        else:
            return get_new_model()
    if model_assets is None:
        return get_new_model()
    else:
        return model_assets


def get_train_data_next_iter(train_data, data_generated, add_old_dataset=False, keep_portion=1.0, **kwargs):
    # **kwargs: a hack of not receiving batch_size from input
    return utils.data_utils.get_train_data_next_iter(train_data, data_generated, add_old_dataset=add_old_dataset,
                                                     keep_portion=keep_portion, batch_size=BATCH_SIZE,
                                                     num_workers=NUM_WORKERS)


def gen_data(model_assets, gen_batch_size, gen_num_batch, **kwargs):
    data_all = []
    model, optimizer = model_assets
    model.eval()
    with torch.no_grad():
        for _ in range(gen_num_batch):
            sample = torch.randn(gen_batch_size, 20).to(device)
            sample = model.decode(sample).cpu()
            data_all.append(sample)
    data_all = torch.cat(data_all, dim=0)
    return data_all


def save_sample(model_assets, log_dir, iteration, transform, **kwargs):
    model, optimizer = model_assets
    with torch.no_grad():
        sample = model.decode(SAMPLE_Z).cpu()
    sample = utils.data_utils.denormalize(sample, transform)
    save_image(sample.view(SAMPLE_NUM, 1, 32, 32), f'{log_dir}/sample_iter_{iteration}' + '.png')


def get_linear_probe_model(model_assets):
    model, optimizer = model_assets
    model.eval()

    def get_latent_fn(enc_model, x):
        mu, logvar = enc_model.encoder(x.view(-1, 1, 32, 32))
        return mu.reshape(mu.shape[0], -1)

    linear_probe_model = LinearProbeModel(model, input_dim=model.latent_dim, output_dim=10,
                                          get_latent_fn=get_latent_fn)
    return linear_probe_model.to(device)
