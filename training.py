# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:37:55 2022

@author: dirty
"""

import pandas as pd
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import PriceMatchDataset
from model import PriceMatchClassifier

def main():

    path = '../data/'
    trainSet = pd.read_csv(path + 'train_processed.csv')
    testSet = pd.read_csv(path + 'test_processed.csv')
    originTestSet = pd.read_csv(path + 'test.csv')
    
    train,vaild = train_test_split(trainSet, test_size=0.3)
    
    # data loader
    train_loader = DataLoader(PriceMatchDataset(train), batch_size = 128, shuffle = False)
    vaild_loader = DataLoader(PriceMatchDataset(vaild), batch_size = 128, shuffle = False)
    
    _,input_dimension = testSet.shape
    model = PriceMatchClassifier(input_dimension)
    
    model.train()
    
    optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=0.01)
    epoch_size = 300
    epoch_pbar = trange(epoch_size, desc="Epoch")
    
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train_loop(train_loader, model, optimizer)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        vaild_loop(vaild_loader, model, optimizer)
        pass

    
    ckpt_path = "../data/checkpoint/checkpoint.pt"
    torch.save(model.state_dict(), str(ckpt_path))
    
def train_loop(dataloader,model,optimizer):
    
    raise NotImplementedError
    
def vaild_loop(dataloader,model,optimizer):
    
    raise NotImplementedError

if __name__ == "__main__":
    main()