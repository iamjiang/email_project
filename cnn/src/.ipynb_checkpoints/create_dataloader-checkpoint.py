import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import sys
import csv
csv.field_size_limit(sys.maxsize)
import torch
from torch.utils.data import Dataset

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
nltk.data.path.append(nltk_data_dir)

class MyDataset(Dataset):
    def __init__(self,data_path,dict_path, max_length):
        super(MyDataset,self).__init__()
        
        dataframe=pd.read_csv(data_path)
        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            next(reader,None) # skip the header
            # for idx, line in enumerate(reader):
            for line in tqdm(reader,total=dataframe.shape[0], leave=True, position=0):
                text = ""
                for tx in line[1].split():
                    text += tx.strip().lower()
                    text += " "
                label = int(line[2])
                texts.append(text)
                labels.append(label)

        self.texts = texts
        self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]     
        self.max_length = max_length
        
    def __len__(self):
        return len(self.labels)        

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        
        document_encode = [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=text)] 
        
        if len(document_encode) < self.max_length:
            extended_docs = [-1 for _ in range(self.max_length - len(document_encode))]
            document_encode.extend(extended_docs)
        else:
            document_encode=document_encode[:self.max_length]

        document_encode = np.array(document_encode)
        document_encode += 1  #### the first row is designated for unk token

        return dict(
            input_ids=document_encode.astype(np.int64), 
            labels=label     
        )
    
    def collate_fn(self, batch):
        input_ids=torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        labels=torch.stack([torch.tensor(x["labels"]) for x in batch])
        
        pad_token_id=0
        keep_mask = input_ids.ne(pad_token_id).any(dim=0)
        input_ids=input_ids[:, keep_mask]
        
        return dict(
            input_ids=input_ids,
            labels=labels
        )        
        

    
    