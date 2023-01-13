import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from models.sparse_attention_bottleneck import SparseAttnBottleneck


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
    def __init__(self, img_size=(1, 32, 32), hid_channels=32, latent_dim=10):
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
        self.lin2 = nn.Linear(hidden_dim, latent_dim)

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

        return x


class DecoderBurgess(nn.Module):
    def __init__(self, img_size=(1, 32, 32), hid_channels=32, latent_dim=10, out_act='sigmoid'):
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

        if out_act == 'sigmoid':
            self.out_act = nn.Sigmoid()
        elif out_act == 'tanh':
            self.out_act = nn.Tanh()

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
        x = self.out_act(self.convT3(x))

        return x


class SparseAttnAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, ch=1, image_size=32, sample_num=100,
                 voc_size=8, proj_hidden_dim=512, out_act='sigmoid', hid_channels=32,
                 num_topk=6, temperature=1.0, loss_fn='mse',
                 bottleneck_use_softmax=True, bottleneck_mirror_neg_code=False):
        """
        Class which defines model and forward pass.
        Parameters
        ----------
        """
        super(SparseAttnAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.hid_channels = hid_channels
        self.num_pixels = image_size * image_size
        self.encoder = EncoderBurgess((ch, image_size, image_size), hid_channels=hid_channels,
                                      latent_dim=self.latent_dim)
        self.decoder = DecoderBurgess((ch, image_size, image_size), hid_channels=hid_channels,
                                      latent_dim=self.latent_dim, out_act=out_act)

        self.loss_fn = loss_fn

        self.ch = ch
        self.image_size = image_size
        self.sample_z = torch.rand(sample_num, ch, image_size, image_size)

        self.voc_size = voc_size
        self.proj_hidden_dim = proj_hidden_dim
        self.proj_output_dim = latent_dim

        self.embedder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim, bias=False),
            nn.BatchNorm1d(self.latent_dim)
        )
        self.sparse_attention = SparseAttnBottleneck(voc_size, self.latent_dim, num_topk=num_topk,
                                                     temperature=temperature, use_softmax=bottleneck_use_softmax,
                                                     mirror_neg_code=bottleneck_mirror_neg_code)

        self.projector = nn.Sequential(
            nn.Linear(self.latent_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, self.proj_output_dim),
        )

        self.reset_parameters()

    def bottleneck(self, latent):
        emb = self.embedder(latent)
        emb_after_attn, top_value, top_ind, dots = self.sparse_attention(emb)
        emb_proj = self.projector(emb_after_attn)
        return emb_proj

    def forward(self, x, **kwargs):
        """
        Forward pass of model.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent = self.encoder(x.view(-1, self.ch, self.image_size, self.image_size))
        latent_sem = self.bottleneck(latent)
        reconstruct = self.decoder(latent_sem)
        return reconstruct

    def encode(self, x):
        return self.encoder(x.view(-1, self.ch, self.image_size, self.image_size))

    def decode(self, latent_sample):
        return self.decoder(latent_sample)

    def reset_parameters(self):
        self.apply(weights_init)


from torchvision import datasets, transforms
import utils.data_utils


def get_data_config(dataset_name):
    if dataset_name in ['mnist', 'fashion_mnist', 'kuzushiji', 'google_draw']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
        ])
        config = {'image_size': 32, 'ch': 1, 'transform': transform, 'out_act': 'sigmoid'}
        return config
    elif dataset_name == 'omniglot':
        transform = transforms.Compose([
            transforms.ToTensor(),
            utils.data_utils.flip_image_value,
            transforms.Resize(32),
        ])
        config = {'image_size': 32, 'ch': 1, 'transform': transform, 'out_act': 'sigmoid'}
        return config
    elif dataset_name in ['cifar10', 'cifar100']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        config = {'image_size': 32, 'ch': 3, 'transform': transform, 'out_act': 'tanh'}
        return config
    elif dataset_name in ['wikiart', 'celeba']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((64, 64)),  # TODO: decide image size
        ])
        config = {'image_size': 64, 'ch': 3, 'transform': transform, 'out_act': 'tanh'}
        return config
    elif dataset_name in ['mpi3d', 'dsprite']:
        transform = None
        config = {'image_size': 64, 'ch': 1, 'transform': transform, 'out_act': 'sigmoid'}
        return config
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_new_model(**kwargs):
    model = SparseAttnAutoencoder(**kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer
