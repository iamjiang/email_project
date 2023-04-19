import time
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src import utils
from src.utils import get_max_lengths
from src.create_dataloader import MyDataset
from src.hierarchical_att_model import HierAttNet
# from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
import random
from tqdm.auto import tqdm

from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score
from sklearn.metrics import classification_report, confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--loss_weight", action='store_true', help="weight for unbalance data")
    parser.add_argument("--train_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--train_negative_positive_ratio", type=int,default=5,help="Undersampling negative vs \
    position ratio in training")
    parser.add_argument("--val_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--val_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs  \
    position ratio in validation")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. \
                        Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="./datasets/train_df.csv")
    parser.add_argument("--val_set", type=str, default="./datasets/val_df.csv")
    parser.add_argument("--val_interval", type=int, default=1, help="Number of epoches between validation phases")
    parser.add_argument("--word2vec_path", type=str, default="/opt/omniai/work/instance1/jupyter/transformers-models/glove/glove.6B.50d.txt")
    # parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="./trained_models")
    parser.add_argument("--seed",  type=int,default=101,help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")
    parser.add_argument("--val_min_recall", default=0.9, type=float, help="minimal recall for valiation dataset")
    parser.add_argument("--word_ratio", default=0.9, type=float, help="ratio to truncate the sentence")
    parser.add_argument("--sent_ratio", default=0.5, type=float, help="ratio to truncate the document")
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train(opt,device):
    seed_everything(opt.seed)
    output_file = open("logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    val_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}

    max_word_length, max_sent_length = get_max_lengths(opt.train_set, word_ratio=opt.word_ratio, sent_ratio=opt.sent_ratio)
    print()
    print('{:<30}{:<10,} '.format("max word length",max_word_length))
    print('{:<30}{:<10,} '.format("max sentence length",max_sent_length))
    print()
    
    train_set = MyDataset(opt.train_set, opt.word2vec_path, max_sent_length, max_word_length)
    train_dataloader = DataLoader(train_set, **training_params)
    val_set = MyDataset(opt.val_set, opt.word2vec_path, max_sent_length, max_word_length)
    val_dataloader = DataLoader(val_set, **val_params)
    print()
    print('{:<30}{:<10,} '.format("train mini-batch",len(train_dataloader)))
    print('{:<30}{:<10,} '.format("validation mini-batch",len(val_dataloader)))
    print()
    model = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, train_set.num_classes,
                       opt.word2vec_path, max_sent_length, max_word_length)


    # if os.path.isdir(opt.log_path):
    #     shutil.rmtree(opt.log_path)
    # os.makedirs(opt.log_path)
    
    print(f"The # of parameters to be updated : {sum([p.nelement() for p in model.parameters() if p.requires_grad==True]):,}")

    # if torch.cuda.is_available():
    model=model.to(device)
    
    train_df=pd.read_csv(opt.train_set)
    train_label=train_df['target'].values.squeeze()
    num_classes=np.unique(train_label).shape[0]
    if opt.loss_weight:
        train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
        loss_weight=torch.tensor(train_classes_weight).to(device)
    else:
        loss_weight=None
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    best_metrics = float('inf')
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(train_dataloader)
    
    iter_tput = []
    for epoch in tqdm(range(opt.num_epoches), position=0, leave=True):
        for step, (feature, label) in enumerate(train_dataloader):
            model.train()
            t0=time.time()
            feature = feature.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            model._init_hidden_state()
            predictions = model(feature)

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


        val_pred,val_target,val_losses=utils.eval_func(val_dataloader,
                                                       model, 
                                                       device,
                                                       num_classes=num_classes, 
                                                       loss_weight=loss_weight)
        
        model.train()
        if not os.path.exists(opt.saved_path):
            os.makedirs(opt.saved_path)

        avg_val_loss=np.mean(val_losses)

        selected_metrics=avg_val_loss
        if selected_metrics + opt.es_min_delta<best_metrics:
            best_metrics=selected_metrics
            best_epoch = epoch
            torch.save(model, opt.saved_path + os.sep + "whole_model_han")
            
        #Early stopping
        if epoch - best_epoch > opt.es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
            break        
 
    model = torch.load(opt.saved_path + os.sep + "whole_model_han")
    val_pred,val_target,val_losses=utils.eval_func(val_dataloader,
                                               model, 
                                               device,
                                               num_classes=num_classes, 
                                               loss_weight=loss_weight)
    
    best_threshold=utils.find_optimal_threshold(val_target.squeeze(), val_pred[:,1].squeeze(), min_recall=opt.val_min_recall, pos_label=False)
    
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
                
if __name__ == "__main__":
    opt = get_args()
    print()
    print(opt)
    print()
    
    data_dir="./datasets"
    if opt.train_undersampling:
        train_df=pd.read_csv(os.path.join("./datasets","train_df.csv"))
        train_df=train_df.loc[:,["preprocessed_email","target"]]
        sample_train=utils.under_sampling(train_df,"target",opt.seed,opt.train_negative_positive_ratio)
        sample_train.to_csv(os.path.join("./datasets","sample_train.csv"))
        opt.train_set=os.path.join("./datasets","sample_train.csv")
        
    if opt.val_undersampling:
        val_df=pd.read_csv(os.path.join("./datasets","val_df.csv"))
        val_df=val_df.loc[:,["preprocessed_email","target"]]
        sample_val=utils.under_sampling(val_df,"target",opt.seed,opt.val_negative_positive_ratio)
        sample_val.to_csv(os.path.join("./datasets","sample_val.csv"))
        opt.val_set=os.path.join("./datasets","sample_val.csv")        

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train(opt,device)
