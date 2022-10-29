# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:37:25 2022

@author: dirty
"""

from typing import Dict

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PriceMatchClassifier(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_class: int,
        intput_size:int,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_class = num_class
        self.intput_size = intput_size
        
        super(PriceMatchClassifier, self).__init__()
        
        # TODO: model architecture
        # CNN

        #self.linear = torch.nn.Linear(in_features=hidden_size*2, out_features=self.num_class)
        self.seq = torch.nn.Sequential(
            
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512*2*128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_class),
            #torch.nn.ReLU(),
            torch.nn.Softmax(dim=1),
            
        )

        
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        
        result = self.seq(batch)

        return result

