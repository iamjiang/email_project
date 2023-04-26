import sys
import csv
csv.field_size_limit(sys.maxsize)
import argparse
import os
import logging
import math
import time
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
import shutil
import time
import datetime
import random
import math
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score

from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk

# os.system("python -m spacy download en_core_web_md")
import spacy
model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","en_core_web_md","en_core_web_md-3.3.0")
nlp = spacy.load(model_name)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler

import src.utils as utils
from src.cnn_model import CNN_Model
from src.create_dataloader import MyDataset

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
       
def main(args, device):
    
    seed_everything(args.seed)
    
    # train_module=MyDataset(args.train_set, args.word2vec_path, args.max_len)
    val_module=MyDataset(args.val_set, args.word2vec_path, args.max_len)
    test_module=MyDataset(args.test_set, args.word2vec_path, args.max_len)

    # train_dataloader=DataLoader(train_module,
    #                             shuffle=True,
    #                             batch_size=args.train_batch_size,
    #                             collate_fn=train_module.collate_fn,
    #                             drop_last=False  
    #                            )

    valid_dataloader=DataLoader(val_module,
                                shuffle=False,
                                batch_size=args.batch_size,
                                collate_fn=val_module.collate_fn
                               )

    test_dataloader=DataLoader(test_module,
                                shuffle=False,
                                batch_size=args.batch_size,
                                collate_fn=test_module.collate_fn
                               )

    # %pdb
    # next(iter(train_dataloader))

    print()
    # print('{:<30}{:<10,} '.format("training mini-batch",len(train_dataloader)))
    print('{:<30}{:<10,} '.format("validation mini-batch",len(valid_dataloader)))
    print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))    
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    model = torch.load(args.pre_trained_model)  
    model=model.to(device)
    
    val_pred,val_target,val_losses=utils.eval_func(valid_dataloader,
                                                   model, 
                                                   device,
                                                   num_classes=2, 
                                                   loss_weight=None)

    test_pred,test_target,test_losses=utils.eval_func(test_dataloader,
                                                      model, 
                                                      device,
                                                      num_classes=2, 
                                                      loss_weight=None)
    
    best_threshold=utils.find_optimal_threshold(val_target.squeeze(), val_pred[:,1].squeeze(), min_recall=args.val_min_recall, pos_label=False)
    
    y_pred=[1 if x>best_threshold else 0 for x in test_pred[:,1]]
    test_output=utils.model_evaluate(test_target.reshape(-1),y_pred)
    
    
    fieldnames = ['True label', 'Predicted label', 'Predicted_prob']
    with open(args.output + os.sep + "predictions.csv", 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(test_target, y_pred, test_pred[:,1]):
            writer.writerow(
                {'True label': i , 'Predicted label': j, 'Predicted_prob': k})    

    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="CNN Model")
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--train_set", type=str, default="./datasets/train_df.csv")
    parser.add_argument("--val_set", type=str, default="./datasets/val_df.csv")
    parser.add_argument("--test_set", type=str, default="./datasets/test_df.csv")
    parser.add_argument("--word2vec_path", type=str, default="/opt/omniai/work/instance1/jupyter/transformers-models/glove/glove.6B.50d.txt")
    parser.add_argument("--pre_trained_model", type=str, default="trained_models/whole_model_cnn")
    parser.add_argument("--train_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--train_negative_positive_ratio", type=int,default=5,help="Undersampling negative vs position ratio in training")
    parser.add_argument("--val_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--val_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs  \
    position ratio in validation")
    parser.add_argument("--output", type=str, default="predictions")    
    parser.add_argument("--val_min_recall", default=0.9, type=float, help="minimal recall for valiation dataset")
    parser.add_argument("--device", default="cpu", type=str)

    # parser.add_argument('--in_channels', type=int, default=1)
    # parser.add_argument('--out_channels', type=int, default=100)
    # parser.add_argument('--kernel_heights', type=int, default=[3,4,5,6], nargs='+', help='kernel size')
    # parser.add_argument('--stride', type=int, default=1)
    # parser.add_argument('--padding', type=int, default=0)
    # parser.add_argument('--mode', type=str, default="non-static",help="mode to update word to vector embedding")
    # parser.add_argument('--keep_probab', type=float, default=0.2)                    
    # parser.add_argument("--max_len", type=int, default=100, help="maximal length of email")
    # parser.add_argument('--train_batch_size', type=int, default=64)
    # parser.add_argument('--val_batch_size', type=int, default=256)
    # parser.add_argument("--num_epoches", type=int, default=100)

    
    args= parser.parse_args()
    
    print()
    print(args)
    print()
      
    data_dir="./datasets"
    if args.train_undersampling:
        train_df=pd.read_csv(os.path.join("./datasets","train_df.csv"))
        train_df=train_df.loc[:,["preprocessed_email","target"]]
        sample_train=utils.under_sampling(train_df,"target",args.seed,args.train_negative_positive_ratio)
        sample_train.to_csv(os.path.join("./datasets","sample_train.csv"))
        args.train_set=os.path.join("./datasets","sample_train.csv")
        
    if args.val_undersampling:
        val_df=pd.read_csv(os.path.join("./datasets","val_df.csv"))
        val_df=val_df.loc[:,["preprocessed_email","target"]]
        sample_val=utils.under_sampling(val_df,"target",args.seed,args.val_negative_positive_ratio)
        sample_val.to_csv(os.path.join("./datasets","sample_val.csv"))
        args.val_set=os.path.join("./datasets","sample_val.csv")        

    max_len = utils.get_max_lengths(args.train_set)
    args.max_len=max_len
    print()
    print('{:<35}{:<10,} '.format("maximal length to truncate email",max_len))
    print()    
    
    device=torch.device(args.device)
    
    main(args, device)
    
