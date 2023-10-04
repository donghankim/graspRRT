import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from models import miniUNet
from datasets import RGBDataset

import os
import img_utils
from random import seed
import math, pdb


if __name__ == "__main__":
    seed(0)
    torch.manual_seed(0)


    # Check if GPU is being detected (added support for m-series mac)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device:", device)

    dataset = RGBDataset("./dataset/")
    train_size = int(len(dataset)*0.9)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    batch_size = 10
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)

    model = miniUNet(3, 4)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0025)

    model_path = "checkpoint_multi.pth.tar"
    epoch, max_epochs = 1, 15
    best_miou = float('-inf')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5)
    while epoch <= max_epochs:
        print('Epoch (', epoch, '/', max_epochs, ')')
        train_loss, train_miou = run(model, train_loader, criterion, is_train = True, optimizer = optimizer)
        test_loss, test_miou = run(model, test_loader, criterion, is_train = False, optimizer = optimizer)
 
        print('Train loss & mIoU: %0.2f %0.2f' % (train_loss, train_miou))
        print('Test loss & mIoU: %0.2f %0.2f' % (test_loss, test_miou))
        print('---------------------------------')
        if test_miou > best_miou:
            best_miou = test_miou 
            save_chkpt(model, epoch, test_miou, model_path)
        epoch += 1
        lr_scheduler.step()
    
    if best_miou >= 0.95:
        model, epoch, best_miou = load_chkpt(model, model_path, device)
        save_prediction(model, test_loader, dataset.pred_dir, device, batch_size)
        print("predictions saved!")



def run(model, loader, criterion, is_train=False, optimizer=None):
    """
        Run forward pass for each sample in the dataloader. Run backward pass and optimize if training.
        Calculate and return mean_epoch_loss and mean_iou
        :param model (torch.nn.module object): miniUNet model object
        :param loader (torch.utils.data.DataLoader object): dataloader 
        :param criterion (torch.nn.module object): Pytorch criterion object
        :param is_train (bool): True if training
        :param optimizer (torch.optim.Optimizer object): Pytorch optimizer object
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    model.train(is_train)
    mean_epoch_loss, mean_iou = 0.0, 0.0
    n_iters = math.ceil(len(loader.dataset)/loader.batch_size)
    data_iter = iter(loader)

    for _ in range(n_iters): 
        batch = next(data_iter)
        img, gt = batch['input'], batch['target']
        X = img.to(device)
        Y = gt.to(device)
        Y_P = model.forward(X)
        
        optimizer.zero_grad()
        loss = criterion(Y_P, Y)
        loss.backward()
        optimizer.step()
        
        mean_iou += sum(iou(Y_P, Y))
        mean_epoch_loss += loss.item()
        
    mean_iou /= len(loader.dataset)
    mean_epoch_loss /= len(loader.dataset)
    
    return mean_epoch_loss, mean_iou






