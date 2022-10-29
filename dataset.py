# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:37:34 2022

@author: dirty
"""

from typing import List, Dict
from torch.utils.data import Dataset


class PriceMatchDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.label_mapping = label_mapping
        self.max_len = max_len

        #self.collate_fn()
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self,inputType: str, Myinput : List[str]):
        # TODO: implement collate_fn
        
        raise NotImplementedError

