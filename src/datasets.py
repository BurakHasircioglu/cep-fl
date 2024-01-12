import numpy as np
import copy
import torch
import matplotlib.image as mpimg
import urllib.request
import zipfile
import os
import pandas as pd
from torchvision import datasets, transforms
from sampling import dist_datasets_iid, dist_datasets_noniid
from options import args_parser
from torch.utils.data import Dataset, TensorDataset, ConcatDataset

class DRDataset(Dataset):
    def __init__(self, data_label, data_dir, transform):
        super().__init__()
        self.data_label = data_label
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, index):
        img_name = self.data_label.id_code[index] + '.png'
        label = self.data_label.diagnosis[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = mpimg.imread(img_path)
        image = (image + 1) * 127.5
        image = image.astype(np.uint8)
        image = self.transform(image)
        return image, label


def get_dataset(args):

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        if args.dataset == 'cifar10':
            """train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)        
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)"""
            transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            # Load the CIFAR10 dataset
            augmented_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                                    download=True, transform=transform_train)
            original_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)
            
            train_dataset = ConcatDataset([augmented_dataset, original_dataset])

            test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                                   download=True, transform=transform_test)
        elif args.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)    

    elif args.dataset == 'mnist' or args.dataset =='fmnist' or args.dataset =='emnist':
        if args.dataset == 'mnist':
            apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

            data_dir = '../data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        elif args.dataset == 'emnist':
            apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

            data_dir = '../data/emnist/'
            train_dataset = datasets.EMNIST(data_dir, train=True, download=True,
                                            split='byclass', transform=apply_transform)
            test_dataset = datasets.EMNIST(data_dir, train=False, download=True,
                                           split='byclass', transform=apply_transform)        

        elif args.dataset == 'fmnist':
            apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

            data_dir = '../data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    
    """if args.sub_dataset_size > 0:
        rnd_indices = np.random.RandomState(seed=0).permutation(len(train_dataset.data))        
        train_dataset.data = train_dataset.data[rnd_indices]
        if torch.is_tensor(train_dataset.targets):
            train_dataset.targets = train_dataset.targets[rnd_indices]    
        else:
            train_dataset.targets = torch.tensor(train_dataset.targets)[rnd_indices]
        train_dataset.data = train_dataset.data[:args.sub_dataset_size]
        train_dataset.targets = train_dataset.targets[:args.sub_dataset_size]
        print("\nThe chosen sub dataset has the following shape:")
        print(train_dataset.data.shape, train_dataset.targets.shape,"\n")  """      

    if args.iid:                   
        user_groups = dist_datasets_iid(train_dataset, args.num_users)         
    else:
        user_groups = dist_datasets_noniid(train_dataset, args.num_users,
                                            num_shards=1000,                                                
                                            unequal=args.unequal)
        
    return train_dataset, test_dataset, user_groups

## For test
if __name__ == '__main__':
    args = args_parser()
    train_dataset, test_dataset, user_groups = get_dataset(args)
