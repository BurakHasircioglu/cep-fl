import os
import copy
import time
import pickle
import numpy as np
import scipy as sp
import torch

from options import args_parser
from logging_results import logging

from opacus import PrivacyEngine
import opacus
from opacus.accountants.analysis import rdp as privacy_analysis

from tqdm import tqdm

MNIST_SIZE = 60000
EMNIST_SIZE = 697932
CIFAR10_SIZE = 50000

if __name__ == '__main__':
    args = args_parser()  
    print(args)
    
    if args.dataset=='mnist':
        dataset_size = MNIST_SIZE
    elif args.dataset=='cifar10':
        dataset_size = CIFAR10_SIZE
    elif args.dataset=='emnist':
        dataset_size = EMNIST_SIZE
    
    size_per_user = dataset_size/args.num_users
    iter_per_epoch = int(np.ceil(size_per_user/args.local_bs))
    total_iterations = iter_per_epoch*args.epochs
    
    print(f"total iterations: {total_iterations}")
    
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    
    effective_std = args.noise_multiplier*args.local_bs*np.sqrt(args.num_users)/args.max_grad_norm
    
    rdp = privacy_analysis.compute_rdp(
                    q=args.local_bs/dataset_size,
                    noise_multiplier=effective_std,
                    steps=total_iterations,
                    orders=alphas,
                )
    
    eps, best_alpha = privacy_analysis.get_privacy_spent(
            orders=alphas, rdp=rdp, delta=args.delta
        )
    
    print(f"epsilon:{eps}, delta:{args.delta}")
    
