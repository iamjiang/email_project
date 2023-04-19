import pandas as pd
from torch.utils.data.dataset import Dataset
import sys
import csv
csv.field_size_limit(sys.maxsize)
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
nltk.data.path.append(nltk_data_dir)
import numpy as np
from tqdm.auto import tqdm

class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, max_length_sentences=50, max_length_word=50):
        super(MyDataset, self).__init__()
        
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
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in sent_tokenize(text=text)]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return document_encode.astype(np.int64), label
    
if __name__ == '__main__':
    test = MyDataset(data_path="../datasets/val_df.csv", dict_path="/opt/omniai/work/instance1/jupyter/transformers-models/glove/glove.6B.50d.txt")
    print (test.__getitem__(index=1)[0].shape)