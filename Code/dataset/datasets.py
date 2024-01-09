from typing import List
from torch.utils.data import Dataset, DataLoader, Subset
from glob import glob
import os
import pathlib
import pandas as pd
import numpy as np
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torch

import bisect

import pickle

import subprocess
from glob import glob
import logging

import contextlib


logger = logging.getLogger("dataset")


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collate_fn_flat_deal(batch):
    out = {}
    for b in batch:
        for k, bv in b.items():
            v = out.get(k, [])
            v.append(bv)
            out[k] = v
            
    for k,v in out.items():
        if k == 'Y':
            v = torch.as_tensor(v, dtype=torch.long)[0]
        else:
            v = torch.as_tensor(v, dtype=torch.float)
        out[k] = v
    return out

class GermanDataset(Dataset):
    def __init__(self, 
                 name: str, 
                 task_type: str,
                 aux_feat_prob: float,
                 use_cuda: bool,
                 seed: int):
        super().__init__()
        self.name = name
        self.type = task_type

        temp_seed(seed)
        
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'german.data-numeric')
        n_feat = 24
        n_aux_feat = 22
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 1000

        data_initial =  pd.read_csv(data_path, sep = "  " , header = None, engine = 'python')
        data_initial.iloc[np.array(data_initial[24].isnull()), 24] = 2.0
        data_shuffled = data_initial.sample(frac = 1) # Randomly shuffling the dataset
        self.label = np.array(data_shuffled[24] == 1)*1
        data = data_shuffled.iloc[: , :24]
        data.insert(0, column='class', value=self.label)
        for i in range(data.shape[0]):
                data.iloc[i,3] = int(data.iloc[i,3].split(" ")[1])
        
        if self.type == "trapezoidal":
                num_chunks = 10
                chunk_size = int(number_of_instances/10)
                self.aux_mask = np.zeros((number_of_instances, n_aux_feat))
                aux_feat_chunk_list = [round((n_feat/num_chunks)*i) - n_base_feat for i in range(1, num_chunks+1)]
                if aux_feat_chunk_list[0] == -1:
                        aux_feat_chunk_list[0] = 0
                aux_feat_chunk_list
                for i in range(num_chunks):
                        self.aux_mask[chunk_size*i:chunk_size*(i+1), :aux_feat_chunk_list[i]] = 1
        elif self.type == "variable_p":
                self.aux_mask = (np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob).astype(float)
        else:
                raise Exception(f"Please choose the type as \"variable_p\" for ", name, " dataset")

        # Data division
        self.n_base_feat = n_base_feat
        self.n_aux_feat = n_aux_feat
        self.n_base_feat = data.shape[1] - 1 - n_aux_feat
        self.Y = np.array(data.iloc[:,:1])
        self.X_base = np.array(data.iloc[:,1:n_base_feat+1])
        self.X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        self.X_aux_new = np.where(self.aux_mask, self.X_aux, 0)
        self.n_classes = 2
    
    def __getitem__(self, index):
        item = { 
            "X_base": self.X_base[index], 
            "X_aux_new": self.X_aux_new[index], 
            "aux_mask": self.aux_mask[index], 
            "Y": self.Y[index], 
            "label": self.label[index],
        }
        return item

    def __len__(self):
        return len(self.Y)

class GermanDataModule(pl.LightningDataModule):
    def __init__(self,
                 name: str = 'german',
                 task_type: str = "variable_p",
                 aux_feat_prob: float = 0.5,
                 num_workers: int = 5,
                 persistent_workers: bool = True,
                 use_cuda: bool = True,
                 batch_size: int = 1,
                 seed: int = 0
                ):
        super().__init__()
        self.name = name
        self.task_type = task_type
        self.aux_feat_prob = aux_feat_prob
        self.use_cuda = use_cuda
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.batch_size = batch_size
        self.n_classes = 2
        self.seed = seed

    def setup(self, stage=None):
        self.dataset = GermanDataset(name=self.name, 
                                     task_type=self.task_type,
                                     aux_feat_prob=self.aux_feat_prob,
                                     use_cuda=self.use_cuda,
                                     seed=self.seed)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                          shuffle=True, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)


class SvmGuideDataset(Dataset):
    def __init__(self, 
                 name: str, 
                 task_type: str,
                 aux_feat_prob: float,
                 use_cuda: bool,
                 seed: int):
        super().__init__()
        self.name = name
        self.type = task_type

        temp_seed(seed)
        
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'svmguide3.txt')
        n_feat = 21
        n_aux_feat = 19
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 1243

        # reading csv files
        # data_initial =  arff.loadarff(data_path)
        data_initial =  pd.read_csv(data_path, sep=" ", header=None)
        data_initial = data_initial.iloc[:, :22]
        for j in range(1, data_initial.shape[1]):
                for i in range(data_initial.shape[0]):
                        data_initial.iloc[i, j] = data_initial.iloc[i, j].split(":")[1]
        for i in range(data_initial.shape[0]):
                data_initial.iloc[i, 0] = (data_initial.iloc[i, 0] == -1)*1
        data = data_initial.sample(frac = 1)
        label = np.asarray(data[0])

        # Masking
        if type == "trapezoidal":
                num_chunks = 10
                chunk_size = int(number_of_instances/10)
                aux_mask = np.zeros((number_of_instances, n_aux_feat))
                aux_feat_chunk_list = [round((n_feat/num_chunks)*i) - n_base_feat for i in range(1, num_chunks+1)]
                if aux_feat_chunk_list[0] == -1:
                        aux_feat_chunk_list[0] = 0
                aux_feat_chunk_list
                for i in range(num_chunks):
                        aux_mask[chunk_size*i:chunk_size*(i+1), :aux_feat_chunk_list[i]] = 1
        elif type == "variable_p":
                aux_mask = (np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob).astype(float)
        else:
                raise Exception("Please choose the type as \"variable_p\" for ", name, " dataset")

        # Data division
        self.n_base_feat = n_base_feat
        self.n_aux_feat = n_aux_feat
        self.n_base_feat = data.shape[1] - 1 - n_aux_feat
        self.Y = np.array(data.iloc[:,:1])
        self.X_base = np.array(data.iloc[:,1:n_base_feat+1])
        self.X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        self.X_aux_new = np.where(self.aux_mask, self.X_aux, 0)
        self.n_classes = 2
    
    def __getitem__(self, index):
        item = { 
            "X_base": self.X_base[index], 
            "X_aux_new": self.X_aux_new[index], 
            "aux_mask": self.aux_mask[index], 
            "Y": self.Y[index], 
            "label": self.label[index],
        }
        return item

    def __len__(self):
        return len(self.Y)

class SvmGuideDataModule(pl.LightningDataModule):
    def __init__(self,
                 name: str = 'svmguid3',
                 task_type: str = "variable_p",
                 aux_feat_prob: float = 0.5,
                 num_workers: int = 5,
                 persistent_workers: bool = True,
                 use_cuda: bool = True,
                 batch_size: int = 1,
                 seed: int = 0
                ):
        super().__init__()
        self.name = name
        self.task_type = task_type
        self.aux_feat_prob = aux_feat_prob
        self.use_cuda = use_cuda
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.batch_size = batch_size
        self.n_classes = 2
        self.seed = seed

    def setup(self, stage=None):
        self.dataset = SvmGuideDataset(name=self.name, 
                                     task_type=self.task_type,
                                     aux_feat_prob=self.aux_feat_prob,
                                     use_cuda=self.use_cuda,
                                     seed=self.seed)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                          shuffle=True, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)
