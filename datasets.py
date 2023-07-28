import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import random_neq, truncate_pad

class TrainDataset(Dataset):
    def __init__(self, full_data, data, item_n, max_seq_len):
        self.data = data
        self.full_data = full_data
        self.item_n = item_n
        self.max_seq_len = max_seq_len

    def __len__(self):
        #number of users
        return len(self.data)
    
    def __getitem__(self, idx):

        #for any sequence we fins positive and negative samples and truncate as in original
        user = self.data[idx][0]
        original_seq = self.full_data[user][0] #item data for input seq

        pos_ = self.data[idx][1] #pos item
        if pos_ == -1:
            pos_ = np.random.choice(self.full_data[user][1])

        maxlen = self.max_seq_len

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = pos_
        idx = maxlen - 1

        ts = set(original_seq) | set(self.full_data[user][1])
        for i in reversed(original_seq):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, self.item_n + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return {"input_seq": torch.tensor(seq), "pos_seq": torch.tensor(pos), "neg_seq": torch.tensor(neg)} 

class ValidDataset(Dataset):
    
    def __init__(self, full_data, data, item_n, max_seq_len):
        
        self.data = data
        self.item_n = item_n
        self.max_seq_len = max_seq_len
        self.full_data = full_data

    def __len__(self):
        
        #number of users
        return len(self.data)
    
    def __getitem__(self, idx):

        user = self.data[idx][0]
        original_seq = self.full_data[user][0] #item data for input seq
        pos = self.data[idx][1] #pos item

        maxlen = self.max_seq_len

        seq = truncate_pad(original_seq, self.max_seq_len) #truncate or pad the seq

        neg_seq = [] #negatives for metrics 

        #avoiding all the original seq and pos items while choosing negs
        ts = set(self.full_data[user][0]) | set(self.full_data[user][1])

        for i in range(500): #100 negatives per pos
            neg = random_neq(1, self.item_n + 1, ts)
            neg_seq.append(neg)
        
        sample_seq = [pos] + neg_seq

        return {"input_seq": torch.tensor(seq), "sample_seq": torch.tensor(sample_seq)}

class TestDataset(Dataset):
    
    def __init__(self, full_data, data, item_n, max_seq_len):
        
        self.data = data
        self.item_n = item_n
        self.max_seq_len = max_seq_len
        self.full_data = full_data

    def __len__(self):
        
        #number of users
        return len(self.data)
    
    def __getitem__(self, idx):

        user = self.data[idx][0]
        original_seq = self.full_data[user][0] #item data for input seq
        pos = self.data[idx][1] #pos item

        maxlen = self.max_seq_len

        seq = truncate_pad(original_seq, self.max_seq_len) #truncate or pad the seq

        neg_seq = [] #negatives for metrics 

        #avoiding all the original seq and pos items while choosing negs
        ts = set(self.full_data[user][0]) | set(self.full_data[user][1])

        for i in range(500): #100 negatives per pos
            neg = random_neq(1, self.item_n + 1, ts)
            neg_seq.append(neg)
        
        sample_seq = [pos] + neg_seq

        return {"input_seq": torch.tensor(seq), "sample_seq": torch.tensor(sample_seq)}