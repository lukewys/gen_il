import numpy as np
import torch
import torch.utils.data
from utils.train_utils import NUM_WORKERS
from torchvision import datasets, transforms

def get_train_data_next_iter(train_data, data_generated, add_old_dataset=False, keep_portion=1.0,
                             batch_size=128, num_workers=8):
    if add_old_dataset:
        old_data_all = []
        for batch_idx, (data, _) in enumerate(train_data):
            old_data_all.append(data)
        old_data_all = torch.cat(old_data_all, dim=0)
        data_combined = torch.cat([old_data_all, data_generated], dim=0)

        if keep_portion < 1.0:
            data_combined = data_combined[
                np.random.choice(data_combined.shape[0], int(data_combined.shape[0] * keep_portion),
                                 replace=False), ...]

        train_loader_new = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_combined, data_combined),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        return train_loader_new
    else:
        train_loader_new = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_generated, data_generated),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return train_loader_new


def get_init_data(transform, dataset_name, batch_size):
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root='./data/mnist_data/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data/mnist_data/', train=False, transform=transform, download=True)
    elif dataset_name == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(root='./data/fashion_mnist_data/', train=True, transform=transform,
                                              download=True)
        test_dataset = datasets.FashionMNIST(root='./data/fashion_mnist_data/', train=False, transform=transform,
                                             download=True)
    elif dataset_name == 'omniglot':
        train_dataset = datasets.Omniglot(root='./data/omniglot_data/', background=True, transform=transform,
                                          download=True)
        test_dataset = datasets.Omniglot(root='./data/omniglot_data/', background=False, transform=transform,
                                         download=True)
    elif dataset_name == 'kuzushiji':
        train_dataset = datasets.KMNIST(root='./data/kuzushiji_data/', train=True, transform=transform, download=True)
        test_dataset = datasets.KMNIST(root='./data/kuzushiji_data/', train=False, transform=transform, download=True)
    # ------------
    elif dataset_name == 'cifar10':  # TODO: do color image later
        train_dataset = datasets.CIFAR10(root='./data/cifar10_data/', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data/cifar10_data/', train=False, transform=transform, download=True)
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data/cifar100_data/', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR100(root='./data/cifar100_data/', train=False, transform=transform, download=True)
    elif dataset_name == 'wikiart':
        # https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset
        train_dataset = datasets.ImageFolder(root='./data/wikiart_data/train/', transform=transform)
        test_dataset = datasets.ImageFolder(root='./data/wikiart_data/test/', transform=transform)
    else:
        raise Exception('Dataset not supported')

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=NUM_WORKERS)

    return train_loader, test_loader


def flip_image_value(img):
    return torch.abs(img - 1)
