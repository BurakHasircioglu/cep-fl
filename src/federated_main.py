import os
import copy
import time
import pickle
import numpy as np
import scipy as sp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


from options import args_parser
from update import DatasetSplit
from utils import test_inference
from models import LeNet
from utils import average_weights, exp_details, dither_round, grad_to_vec, update_grad
from datasets import get_dataset
from torchvision import models as mdls
from logging_results import logging

from opacus import PrivacyEngine
import opacus
from opacus.data_loader import DPDataLoader
from opacus.validators import ModuleValidator

from tqdm import tqdm

from torch.distributions.gamma import Gamma
from torch.distributions.uniform import Uniform


if __name__ == '__main__':
    
    args = args_parser()  
    print(args)
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'    
    
    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    
    # BUILD MODEL

    if args.dataset == 'mnist':
        global_model = LeNet(num_classes=10)
        global_model.to(device)
    elif args.dataset == 'emnist':
        global_model = LeNet(num_classes=62, use_log_softmax=True) 
        global_model.to(device)
    elif args.dataset == 'cifar10':
        global_model = mdls.resnet18(pretrained=False, num_classes=10)
        global_model.to(device)


    
    if args.withDP or args.withDithering:
        errors = ModuleValidator.validate(global_model, strict=False)
        print(errors[-5:])

        global_model = ModuleValidator.fix(global_model)
        ModuleValidator.validate(global_model, strict=False)
    

    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, 
                                momentum=args.momentum, weight_decay=args.weight_decay) 
    data_loader = DataLoader(train_dataset, batch_size= args.local_bs, shuffle=True)
    
    
    if args.withDP:
        noise_std = args.noise_multiplier*args.local_bs/args.max_grad_norm 
    elif args.withDithering:
        noise_std=0.0 # if dithering, DPOptimizer do not add Gaussian noise

    if args.withDP or args.withDithering:
        pe = PrivacyEngine()
    
        model, optimizer, data_loader = pe.make_private(
            module = global_model,
            optimizer = optimizer,
            data_loader= data_loader,
            noise_multiplier = noise_std,
            max_grad_norm =  args.max_grad_norm,
            )
    else:
        model = global_model
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_period, gamma=args.lr_coeff)

      
    local_trainloaders = []

    for u in range(args.num_users):
        if args.withDP or args.withDithering:
            trainloader = DPDataLoader(dataset=DatasetSplit(train_dataset, user_groups[u]), sample_rate=data_loader.sample_rate) 
        else:
            trainloader = DataLoader(DatasetSplit(train_dataset, user_groups[u]),
                         batch_size= args.local_bs, shuffle=True) 

                      
        local_trainloaders.append(trainloader)
            

    # Training
    criterion = nn.CrossEntropyLoss().to(device)
    train_loss = []
    test_log = []    

    for epoch in range(args.epochs):   
        print(f"epoch: {epoch}, lr: {optimizer.param_groups[0]['lr']}")
        iter_per_epoch = int(np.ceil(len(user_groups[0])/args.local_bs))
        
        local_iterators = []
        for u in range(args.num_users):
            local_iterators.append(iter(local_trainloaders[u]))
        
        iter_loss = []
        for i in tqdm(range(iter_per_epoch)):
            user_loss = []
            average_grads = []
            for u in range(args.num_users):
                torch.cuda.empty_cache()
                
                try:
                    images, labels = next(local_iterators[u])
                except StopIteration:
                    pass
                
                images = images.to(device)
                labels = labels.to(device)
                                
                # Set mode to train model
                model = model.to(device)
                model.train()
                    

                
                optimizer.zero_grad()

                model_preds = model(images)

                    
                loss = criterion(model_preds, labels) #.to(torch.long))
                loss.backward()
                user_loss.append(loss.item())
                
                if args.withDP or args.withDithering:
                    optimizer.pre_step()


                user_grads = grad_to_vec(model)
                
                if args.withDithering:
                    grad_u = user_grads
                    gamma_dist = Gamma(3/2, 1/2)

                    # Generate samples from the gamma distribution
                    samples = gamma_dist.sample(grad_u.shape)
                    samples = samples.to(device)
                    
                    delta =  2*args.noise_multiplier*torch.sqrt(samples)
                    unif_dist = Uniform(low = -delta/2, high=delta/2)
                    dither_noise = unif_dist.sample().to(device)
                    
                    # Update sent from the client
                    quantized_grad_u = dither_round(grad_u-dither_noise, delta)
                    quantized_grad_u = quantized_grad_u*delta + delta/2
                    
                    # Update received by the PS
                    sum_update = (quantized_grad_u + dither_noise)/args.num_users
                else:
                    sum_update = user_grads/args.num_users
                
                if u==0:
                    average_grads = sum_update
                else:
                    average_grads += sum_update
                   
                optimizer.zero_grad()
            
            update_grad(model, average_grads)
            if args.withDP or args.withDithering:
                optimizer.original_optimizer.step()
            else:
                optimizer.step()

            iter_loss.append(sum(user_loss)/len(user_loss))
        
        train_loss.append(sum(iter_loss)/len(iter_loss))
        scheduler.step()

                
                
        if epoch%(args.testing_period)==0:
            torch.cuda.empty_cache()   

            save_path = 'models/'+args.exp_name+'_'+str(epoch)+'.pth'       
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save(model.state_dict(), save_path)
            _acc, _loss = test_inference(args, model, test_dataset)        
            test_log.append([_acc, _loss])  
          
  
            logging(args, epoch, train_loss, test_log)
