import pandas as pd
import numpy as np
import time
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler

import transformers

class Hierarchical_Model(nn.Module):

    def __init__(self, model, config, pooling_method="mean", num_classes=2):
        super(Hierarchical_Model, self).__init__()

        self.pooling_method = pooling_method
        self.model=model
        self.config=config
        if self.pooling_method=="lstm":
            self.lstm=nn.LSTM(self.config.hidden_size,self.config.hidden_size//4, num_layers=1, bidirectional=False)
            self.out=nn.Linear(self.config.hidden_size//4, num_classes)
        else:
            self.out = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, ids, mask,  lengt):

        output = self.model(ids, attention_mask=mask)
        pooled_output=output['pooler_output']

        chunks_emb = pooled_output.split_with_sizes(lengt.tolist())
        
        if self.pooling_method=="lstm":
            seq_lengths = torch.LongTensor([x for x in map(len, chunks_emb)])
            batch_emb_pad = nn.utils.rnn.pad_sequence(chunks_emb, padding_value=-91, batch_first=True)
            batch_emb = batch_emb_pad.transpose(0, 1)  # (B,L,D) -> (L,B,D)
            lstm_input = nn.utils.rnn.pack_padded_sequence(batch_emb, seq_lengths.cpu().numpy(), batch_first=False, enforce_sorted=False)
            packed_output, (h_t, h_c) = self.lstm(lstm_input, )  # (h_t, h_c))
            emb_pool = h_t.view(-1, self.config.hidden_size//4)
            
        if self.pooling_method == "mean":
            emb_pool = torch.stack([torch.mean(x, 0) for x in chunks_emb])
        elif self.pooling_method == "max":
            emb_pool = torch.stack([torch.max(x, 0)[0] for x in chunks_emb])

        return self.out(emb_pool)
    