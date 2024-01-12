import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager




class SampledDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        return image, label




class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), torch.tensor(label)

class LocalUpdate(object):
    def __init__(self, trainloader): ####!!
        self.trainloader = trainloader

    def get_sampled_batch(self):
        for images, labels in self.trainloader:
            return images, labels
 

class GlobalUpdate(object):
    def __init__(self, args, trainloader, model, u_id, idxs, sampling_prob, optimizer, withDP, noise_std=0): ####!!
        self.u_id = u_id
        self.args = args
        self.trainloader = trainloader
        self.model = model
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.sasampling_prob = sampling_prob ####!!
        self.optimizer = optimizer
        self.withDP = withDP
        self.noise_std = noise_std

    def update_local_model(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)
    
    #def get_sampled_batch(self):
    #    for images, labels in self.trainloader:
    #        return images, labels
    #    
    # def update_weights(self):
    #     # Set mode to train model
    #     model = self.model
    #     model.to(self.device)
    #     model.train()
    #     epoch_loss = []
    
    #     for iter in range(self.args.local_ep): 
    #         batch_loss = []
    #         self.optimizer.zero_grad()
    #         if self.args.withDP:
    #             virtual_batch_rate = int(self.args.virtual_batch_size / self.args.local_bs)            
    #         for batch_idx, (images, labels) in enumerate(self.trainloader):
    #             print('batch_idx:'+str(batch_idx))
    #             print(images.size())
                
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             model_preds = model(images)
    #             loss = self.criterion(model_preds, labels)
    #             loss.backward()
                

    #             self.optimizer.step()
    #             self.optimizer.zero_grad()
    #             #############
    #             batch_loss.append(loss.item())
    #             break
    #         epoch_loss.append(sum(batch_loss)/len(batch_loss))
    #     return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights(self, images, labels):
        # Set mode to train model
        model = self.model
        model.to(self.device)
        model.train()
        epoch_loss = []
    
        for iter in range(1): #range(self.args.local_ep): 
            batch_loss = []
            self.optimizer.zero_grad()

#            images, labels = images.to(self.device), labels.to(self.device)
            if images.size(0) == 0 and self.withDP:
                self.optimizer.zero_grad()
                self.optimizer.step(zero_samples=True)
                return 0.0 

            received_data = SampledDataset(images, labels)
            temp_loader = torch.utils.data.DataLoader(received_data, batch_size = images.size(0), shuffle=True)
            if self.withDP:
                with BatchMemoryManager(data_loader=temp_loader, max_physical_batch_size=self.args.virtual_batch_size, optimizer=self.optimizer
                        ) as new_data_loader:
                    for im, lab in new_data_loader:
                        model_preds = model(im)
    #                    print(model_preds.dtype)
    #                    print(model_preds.size())
    #                    print(lab.dtype)
    #                    print(lab)
    #                    print(lab.size())
                        loss = self.criterion(model_preds, lab.to(torch.long))
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            else:
                for im, lab in temp_loader:
                    model_preds = model(im)
                    loss = self.criterion(model_preds, lab.to(torch.long))
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            #############
            batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)
