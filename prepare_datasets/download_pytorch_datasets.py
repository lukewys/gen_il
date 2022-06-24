from torchvision import datasets, transforms

transform = transforms.Compose([])
# train_dataset = datasets.MNIST(root='../data/mnist_data/', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='../data/mnist_data/', train=False, transform=transform, download=True)
# train_dataset = datasets.FashionMNIST(root='../data/fashion_mnist_data/', train=True, transform=transform,
#                                       download=True)
# test_dataset = datasets.FashionMNIST(root='../data/fashion_mnist_data/', train=False, transform=transform,
#                                      download=True)
# train_dataset = datasets.Omniglot(root='../data/omniglot_data/', background=True, transform=transform,
#                                   download=True)
# test_dataset = datasets.Omniglot(root='../data/omniglot_data/', background=False, transform=transform,
#                                  download=True)
# train_dataset = datasets.KMNIST(root='../data/kuzushiji_data/', train=True, transform=transform, download=True)
# test_dataset = datasets.KMNIST(root='../data/kuzushiji_data/', train=False, transform=transform, download=True)
# train_dataset = datasets.CIFAR10(root='../data/cifar10_data/', train=True, transform=transform, download=True)
# test_dataset = datasets.CIFAR10(root='../data/cifar10_data/', train=False, transform=transform, download=True)
train_dataset = datasets.CelebA(root='../data/celeba_data/', split='train', transform=transform, download=True)
test_dataset = datasets.CelebA(root='../data/celeba_data/', split='test', transform=transform, download=True)
# train_dataset = datasets.CIFAR100(root=f'../data/cifar100_data/', train=True, transform=transform,
#                                   download=True)
# test_dataset = datasets.CIFAR100(root=f'../data/cifar100_data/', train=False, transform=transform,
#                                  download=True)
