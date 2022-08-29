from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

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


def get_init_data(transform, dataset_name, batch_size, data_dir='./data'):
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root=f'{data_dir}/mnist_data/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root=f'{data_dir}/mnist_data/', train=False, transform=transform, download=True)
    elif dataset_name == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(root=f'{data_dir}/fashion_mnist_data/', train=True, transform=transform,
                                              download=True)
        test_dataset = datasets.FashionMNIST(root=f'{data_dir}/fashion_mnist_data/', train=False, transform=transform,
                                             download=True)
    elif dataset_name == 'omniglot':
        train_dataset = datasets.Omniglot(root=f'{data_dir}/omniglot_data/', background=True, transform=transform,
                                          download=True)
        test_dataset = datasets.Omniglot(root=f'{data_dir}/omniglot_data/', background=False, transform=transform,
                                         download=True)
    elif dataset_name == 'kuzushiji':
        train_dataset = datasets.KMNIST(root=f'{data_dir}/kuzushiji_data/', train=True, transform=transform,
                                        download=True)
        test_dataset = datasets.KMNIST(root=f'{data_dir}/kuzushiji_data/', train=False, transform=transform,
                                       download=True)
    # ------------
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=f'{data_dir}/cifar10_data/', train=True, transform=transform,
                                         download=True)
        test_dataset = datasets.CIFAR10(root=f'{data_dir}/cifar10_data/', train=False, transform=transform,
                                        download=True)
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root=f'{data_dir}/cifar100_data/', train=True, transform=transform,
                                          download=True)
        test_dataset = datasets.CIFAR100(root=f'{data_dir}/cifar100_data/', train=False, transform=transform,
                                         download=True)
    elif dataset_name == 'wikiart':
        # https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset
        train_dataset = datasets.ImageFolder(root=f'{data_dir}/wikiart_split/train/', transform=transform)
        test_dataset = datasets.ImageFolder(root=f'{data_dir}/wikiart_split/test/', transform=transform)
    elif dataset_name == 'celeba':
        train_dataset = datasets.CelebA(root=f'{data_dir}/celeba_data/', split='train', transform=transform,
                                        download=True)
        test_dataset = datasets.CelebA(root=f'{data_dir}/celeba_data/', split='test', transform=transform,
                                       download=True)
    elif dataset_name == 'dsprite':
        # TODO: split according to OOD generalization
        # (737280, 64, 64)
        data = np.load(f'{data_dir}/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')['imgs'].astype(
            np.float32) / 255

        # (100000, 1, 64, 64)
        train_data = torch.from_numpy(data[:100000]).unsqueeze(1).float()
        # (30000, 1, 64, 64)
        test_data = torch.from_numpy(data[100000:100000 + 30000]).unsqueeze(1).float()

        train_dataset = torch.utils.data.TensorDataset(train_data, train_data)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_data)
    elif dataset_name == 'mpi3d':
        # (100000, 64, 64, 3)
        train_data = torch.from_numpy(
            np.load(f'{data_dir}/mpi3d/mpi3d_realistic_subset_train.npy').astype(np.float32) / 255 * 2 - 1)
        # (30000, 64, 64, 3)
        test_data = torch.from_numpy(
            np.load(f'{data_dir}/mpi3d/mpi3d_realistic_subset_test.npy').astype(np.float32) / 255 * 2 - 1)

        train_data = train_data.permute(0, 3, 1, 2)
        test_data = test_data.permute(0, 3, 1, 2)

        train_dataset = torch.utils.data.TensorDataset(train_data, train_data)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_data)
    elif dataset_name == 'google_draw':
        # (5042617, 1, 28, 28)
        train_data = torch.from_numpy(
            np.load(f'{data_dir}/google_draw/google_draw_train.npy').astype(np.float32) / 255).reshape(-1, 1, 28, 28)
        # (504098, 1, 28, 28)
        test_data = torch.from_numpy(
            np.load(f'{data_dir}/google_draw/google_draw_test.npy').astype(np.float32) / 255).reshape(-1, 1, 28, 28)
        train_dataset = torch.utils.data.TensorDataset(train_data, train_data)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_data)
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


def denormalize(x, transform):
    # https://github.com/pytorch/vision/issues/848
    # x: B, C, H, W
    mean = None
    std = None
    for t in transform.transforms:
        if 'Normalize' in str(t):
            mean = t.mean
            std = t.std
    if mean is None or std is None:
        return x
    # C, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)
