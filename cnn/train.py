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
    
    train_module=MyDataset(args.train_set, args.word2vec_path, args.max_len)
    val_module=MyDataset(args.val_set, args.word2vec_path, args.max_len)


    train_dataloader=DataLoader(train_module,
                                shuffle=True,
                                batch_size=args.train_batch_size,
                                collate_fn=train_module.collate_fn,
                                drop_last=False ,
                                pin_memory=True
                               )

    valid_dataloader=DataLoader(val_module,
                                shuffle=False,
                                batch_size=args.val_batch_size,
                                collate_fn=val_module.collate_fn,
                                drop_last=False,
                                pin_memory=True
                               )

    # test_dataloader=DataLoader(test_module,
    #                             shuffle=False,
    #                             batch_size=args.test_batch_size,
    #                             collate_fn=test_module.collate_fn
    #                            )

    # %pdb
    # next(iter(train_dataloader))

    print()
    print('{:<30}{:<10,} '.format("training mini-batch",len(train_dataloader)))
    print('{:<30}{:<10,} '.format("validation mini-batch",len(valid_dataloader)))
    # print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))    

    dict = pd.read_csv(filepath_or_buffer=args.word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    dict_len, embed_size = dict.shape
    dict_len += 1
    vocab_size=dict_len
    embedding_length=embed_size
    
    unknown_word = np.zeros((1, embed_size))
    pretrained_emb = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
    
    train_df=pd.read_csv(args.train_set)
    train_label=train_df['target'].values.squeeze()
    num_classes=np.unique(train_label).shape[0]
    if args.loss_weight:
        train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
        loss_weight=torch.tensor(train_classes_weight).to(device)
    else:
        loss_weight=None

    model=CNN_Model(args.train_batch_size, num_classes, args.in_channels, args.out_channels, args.kernel_heights, args.stride, args.padding, \
                    args.keep_probab, vocab_size, embedding_length, pretrained_emb, args.mode)
    model.to(device)
    
    print()
    print(f"The # of parameters to be updated : {sum([p.nelement() for p in model.parameters() if p.requires_grad==True]):,}")
    print()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum)
    best_metrics = float('inf')
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(train_dataloader)
    
    iter_tput = []
    for epoch in tqdm(range(args.num_epoches), position=0, leave=True):
        for step, batch in enumerate(train_dataloader):
            model.train()
            t0=time.time()
            feature = batch["input_ids"].to(device)
            label =  batch["labels"].to(device)
            optimizer.zero_grad()
            predictions = model(feature,device)

            if loss_weight is None:
                loss = F.cross_entropy(predictions.view(-1, num_classes),label)
            else:
                loss = F.cross_entropy(predictions.view(-1, num_classes),label, weight=loss_weight.float()) 
                
            loss.backward()
            optimizer.step()
            
            iter_tput.append(feature.shape[0] / (time.time() - t0))
            
            if step%(len(train_dataloader)//10)==0 and not step==0 :
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.2f} | GPU{:.0f} MB'
                                  .format(epoch, step, loss, np.mean(iter_tput[3:]), 
                                          torch.cuda.max_memory_allocated() / 1000000))


        val_pred,val_target,val_losses=utils.eval_func(valid_dataloader,
                                                       model, 
                                                       device,
                                                       num_classes=num_classes, 
                                                       loss_weight=loss_weight)
        
        model.train()
        if not os.path.exists(args.saved_path):
            os.makedirs(args.saved_path)

        avg_val_loss=np.mean(val_losses)

        selected_metrics=avg_val_loss
        if selected_metrics + args.es_min_delta<best_metrics:
            best_metrics=selected_metrics
            best_epoch = epoch
            torch.save(model, args.saved_path + os.sep + "whole_model_cnn")
            
        #Early stopping
        if epoch - best_epoch > args.es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
            break        
 
    model = torch.load(args.saved_path + os.sep + "whole_model_cnn")
    val_pred,val_target,val_losses=utils.eval_func(valid_dataloader,
                                               model, 
                                               device,
                                               num_classes=num_classes, 
                                               loss_weight=loss_weight)
    
    best_threshold=utils.find_optimal_threshold(val_target.squeeze(), val_pred[:,1].squeeze(), min_recall=args.val_min_recall, pos_label=False)
    
    y_pred=[1 if x>best_threshold else 0 for x in val_pred[:,1]]
    test_output=utils.model_evaluate(val_target.reshape(-1),y_pred)


#     with open(os.path.join(output_dir,"metrics_test.txt"),'a') as f:
#         f.write(f'{args.model_name},{test_output["total positive"]},{test_output["false positive"]},{test_output["false_negative"]}, \
#         {test_output["precision"]},{test_output["recall"]},{test_output["f1_score"]},{test_output["AUC"]},{test_output["pr_auc"]},{best_threshold}\n')  

#     with open(os.path.join(output_dir,"y_true_pred.txt"),'w') as f:
#         for x,y,z in zip(test_target.tolist(),y_pred,test_pred[:,1].tolist()):
#             f.write(str(x)+","+str(y)+","+str(z)+ '\n')

    print("==> performance on test set \n")
    print("")
    print("total positive: {:,} | false positive: {:,} | false_negative: {:,} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%}".format(test_output["total positive"], test_output["false positive"], test_output["false_negative"], \
                 test_output["precision"], test_output["recall"], test_output["f1_score"], test_output["AUC"], test_output["pr_auc"]))

    print()
    print(f"\n===========Test Set Performance===============\n")
    print()

    print(classification_report(val_target, y_pred))
    print()
    print(confusion_matrix(val_target, y_pred)) 
    
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="CNN Model")
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument("--train_set", type=str, default="./datasets/train_df.csv")
    parser.add_argument("--val_set", type=str, default="./datasets/val_df.csv")
    parser.add_argument("--word2vec_path", type=str, default="/opt/omniai/work/instance1/jupyter/transformers-models/glove/glove.6B.50d.txt")
    parser.add_argument("--loss_weight", action='store_true', help="weight for unbalance data")
    parser.add_argument("--train_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--train_negative_positive_ratio", type=int,default=5,help="Undersampling negative vs position ratio in training")
    parser.add_argument("--val_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--val_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs  \
    position ratio in validation")
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=3,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. \
                        Set to 0 to disable this technique.")
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=100)
    parser.add_argument('--kernel_heights', type=int, default=[3,4,5], nargs='+', help='kernel size')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--mode', type=str, default="non-static",help="mode to update word to vector embedding")
    parser.add_argument('--keep_probab', type=float, default=0.2)                    
    parser.add_argument("--max_len", type=int, default=100, help="maximal length of email")
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument("--num_epoches", type=int, default=10)
    parser.add_argument("--val_min_recall", default=0.9, type=float, help="minimal recall for valiation dataset")
    parser.add_argument("--saved_path", type=str, default="./trained_models")
    parser.add_argument("--device", default="cpu", type=str)
    
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
    
