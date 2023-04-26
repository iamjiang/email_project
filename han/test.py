import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src import utils
from src.create_dataloader import MyDataset
import argparse
import shutil
import csv
import numpy as np
import random
from tqdm.auto import tqdm

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    
    parser.add_argument("--batch_size", type=int, default=1024)
    # parser.add_argument("--val_set", type=str, default="./datasets/val_df.csv")
    # parser.add_argument("--test_set", type=str, default="./datasets/test_df.csv")
    parser.add_argument("--val_set", type=str, default="./datasets/sample_val.csv")
    parser.add_argument("--test_set", type=str, default="./datasets/sample_test.csv")
    parser.add_argument("--pre_trained_model", type=str, default="trained_models/whole_model_han")
    parser.add_argument("--word2vec_path", type=str, default="/opt/omniai/work/instance1/jupyter/transformers-models/glove/glove.6B.50d.txt")
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--seed",  type=int,default=101,help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")
    parser.add_argument("--val_min_recall", default=0.9, type=float, help="minimal recall for valiation dataset")
    # parser.add_argument("--word_ratio", default=0.9, type=float, help="ratio to truncate the sentence")
    # parser.add_argument("--sent_ratio", default=0.5, type=float, help="ratio to truncate the document")
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
    
    
def test(opt, device):
    
    seed_everything(opt.seed)
    
    val_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    
    test_params = {"batch_size": opt.batch_size,
               "shuffle": False,
               "drop_last": False}
    
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
        
    model = torch.load(opt.pre_trained_model)  
    model=model.to(device)
    
    val_set = MyDataset(opt.val_set, opt.word2vec_path, model.max_sent_length, model.max_word_length)
    val_dataloader = DataLoader(val_set, **val_params)
    
    test_set = MyDataset(opt.val_set, opt.word2vec_path, model.max_sent_length, model.max_word_length)
    test_dataloader = DataLoader(test_set, **test_params)
    
    print()
    print('{:<30}{:<10,} '.format("validation mini-batch",len(val_dataloader)))
    print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))
    print()    
    
    
    val_pred,val_target,val_losses=utils.eval_func(val_dataloader,
                                                   model, 
                                                   device,
                                                   num_classes=2, 
                                                   loss_weight=None)

    test_pred,test_target,test_losses=utils.eval_func(test_dataloader,
                                                      model, 
                                                      device,
                                                      num_classes=2, 
                                                      loss_weight=None)
    
    best_threshold=utils.find_optimal_threshold(val_target.squeeze(), val_pred[:,1].squeeze(), min_recall=opt.val_min_recall, pos_label=False)
    
    y_pred=[1 if x>best_threshold else 0 for x in test_pred[:,1]]
    test_output=utils.model_evaluate(test_target.reshape(-1),y_pred)
    
    
    fieldnames = ['True label', 'Predicted label', 'Predicted_prob']
    with open(opt.output + os.sep + "predictions.csv", 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(test_target, y_pred, test_pred[:,1]):
            writer.writerow(
                {'True label': i , 'Predicted label': j, 'Predicted_prob': k})    
    
    
if __name__ == "__main__":
    opt = get_args()
    print()
    print(opt)
    print()
         

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test(opt,device)

#     with open(os.path.join(output_dir,"metrics_test.txt"),'a') as f:
#         f.write(f'{args.model_name},{test_output["total positive"]},{test_output["false positive"]},{test_output["false_negative"]}, \
#         {test_output["precision"]},{test_output["recall"]},{test_output["f1_score"]},{test_output["AUC"]},{test_output["pr_auc"]},{best_threshold}\n')  

#     with open(os.path.join(output_dir,"y_true_pred.txt"),'w') as f:
#         for x,y,z in zip(test_target.tolist(),y_pred,test_pred[:,1].tolist()):
#             f.write(str(x)+","+str(y)+","+str(z)+ '\n')