import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

import datasets
from datasets import load_dataset, load_from_disk,load_metric, concatenate_datasets, DatasetDict, Dataset
from datasets import disable_caching, enable_caching
enable_caching()
from transformers import AutoModelForMaskedLM , AutoTokenizer, AutoConfig
from transformers import AdamW, get_scheduler
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import argparse
import model_prompt 
import wrapper
import utils

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(args, wrapped_text, device, prompt, token_list1=["positive"], token_list2=["negative"]):
    
    # train_df=Dataset.from_pandas(train_data)
    # val_df=Dataset.from_pandas(val_data)
    # test_df=Dataset.from_pandas(test_data)
    
    model_path=args.model_name.split("-")[0]+"_"+args.model_name.split("-")[1]+"_"+"repo"
    if args.customized_tokenizer:
        model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints/few-shot/prompt-learning/fine-tune-LM",model_path,"JPMC-email-tokenizer")
    else:
        model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints/few-shot/prompt-learning/fine-tune-LM",model_path)
        
    prompting= model_prompt.Prompting(model_name)
    
    benchmark=prompting.benchmark_prompt(prompt, token_list1, token_list2, device)
    prob=torch.nn.functional.softmax(benchmark.unsqueeze(0), dim=1)
    print("{:<60}{:<20.0%}".format("The probability of negative sentiment for prompt only text: ",prob.squeeze()[1].item()))
    print("{:<60}{:<20.0%}".format("The probability of positive sentiment for prompt only text: ",prob.squeeze()[0].item()))
    
    tokenizer=prompting.tokenizer
    model=prompting.model
    
    train_module=wrapper.Loader_Creation(wrapped_text["train"], tokenizer,'wrapped_email')
    val_module=wrapper.Loader_Creation(wrapped_text["validation"], tokenizer,'wrapped_email')
    test_module=wrapper.Loader_Creation(wrapped_text["test"], tokenizer,'wrapped_email')

    train_dataloader=DataLoader(train_module,
                                shuffle=False,
                                batch_size=args.train_batch_size,
                                collate_fn=train_module.collate_fn,
                                drop_last=False   # longformer model bug
                               )

    val_dataloader=DataLoader(val_module,
                                shuffle=False,
                                batch_size=args.train_batch_size,
                                collate_fn=train_module.collate_fn,
                                drop_last=False   # longformer model bug
                               )

    test_dataloader=DataLoader(test_module,
                                shuffle=False,
                                batch_size=args.test_batch_size,
                                collate_fn=train_module.collate_fn,
                                drop_last=False   # longformer model bug
                               )
    
    print()
    print('{:<30}{:<10,} '.format("training mini-batch",len(train_dataloader)))
    print('{:<30}{:<10,} '.format("validation mini-batch",len(val_dataloader)))
    print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))

    train_label=np.array(wrapped_text["train"]['target']).squeeze()
    num_classes=np.unique(train_label).shape[0]
    if args.loss_weight:
        train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
        loss_weight=torch.tensor(train_classes_weight).to(device)
        class_weights=class_weight.compute_class_weight(class_weight="balanced",classes=np.unique(train_label),y=wrapped_text["train"]['target'])
        class_weights=torch.tensor(class_weights).to(device)
        assert torch.all(torch.eq(loss_weight,class_weights))

    else:
        loss_weight=None
        class_weights=None

    t_total = int((len(train_dataloader) // args.train_batch_size)//args.gradient_accumulation_steps*float(args.num_epochs))

    warmup_steps=int((len(train_dataloader) // args.train_batch_size)//args.gradient_accumulation_steps*args.warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, 
                                     optimizer=optimizer,
                                     num_warmup_steps=warmup_steps,
                                     num_training_steps=t_total)

    goodToken=token_list1[0]
    badToken=token_list2[0]
    good=prompting.tokenizer.convert_tokens_to_ids(goodToken)
    bad=prompting.tokenizer.convert_tokens_to_ids(badToken)

    best_metric = float('inf')
    # best_metric = 0

    iter_tput = []
    for epoch in tqdm(range(args.num_epochs),position=0 ,leave=True):
        model.train()
        model=model.to(device)
        losses=[]
        for step , batch in enumerate(train_dataloader):
            t0=time.time()
            # mask_pos=[v.tolist().index(tokenizer.mask_token_id) for v in batch["input_ids"]]
            mask_pos=[v.tolist().index(tokenizer.mask_token_id) if tokenizer.mask_token_id in v.tolist() else len(v) for v in batch["input_ids"]]
            mask_pos=torch.tensor(mask_pos).to(device)
            batch={k:v.type(torch.LongTensor).to(device) for k,v in batch.items()}
            outputs=model(batch['input_ids'])
            predictions = outputs[0]
            pred=predictions.gather(1, mask_pos.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,prompting.tokenizer.vocab_size)).squeeze(1) ## dim=batch_size * vocab_size
            logits=pred[:,[good,bad]] ## dim=batch_size * 2
            prob=torch.nn.functional.softmax(logits, dim=1)
            logits=logits/(benchmark.expand_as(logits).to(device))  #### normalized with benchmark
            if loss_weight is None:
                loss = F.cross_entropy(logits.view(-1, num_classes).to(device),batch["labels"])
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes).to(device),batch["labels"], \
                                       weight=loss_weight.float().to(device)) 

            loss.backward()
            optimizer.step()

            if (step+1)%args.gradient_accumulation_steps == 0 or step==len(train_dataloader)-1:
                optimizer.step()
                if args.use_schedule:
                    lr_scheduler.step()
                optimizer.zero_grad()

            losses.append(loss.item())
            iter_tput.append(batch["input_ids"].shape[0] / (time.time() - t0)) 

            if step%(len(train_dataloader)//10)==0 and not step==0 :
                    print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.2f} | GPU{:.0f} MB'
                          .format(epoch, step, np.mean(losses[-10:]), np.mean(iter_tput[3:]), torch.cuda.max_memory_allocated() / 1000000))

        val_pred,val_target,val_losses=utils.eval_func(val_dataloader,
                                                       model, 
                                                       tokenizer,
                                                       device,
                                                       num_classes=num_classes, 
                                                       loss_weight=loss_weight,
                                                       goodToken=goodToken,
                                                       badToken=badToken)


        output_dir=os.path.join(os.getcwd(), args.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)        
        avg_val_loss=np.mean(val_losses)
        selected_metric=avg_val_loss
        if selected_metric<best_metric:
            best_metric=selected_metric
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    config=AutoConfig.from_pretrained(output_dir)
    model = AutoModelForMaskedLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)


    val_pred,val_target,val_losses=utils.eval_func(val_dataloader,
                                                   model, 
                                                   tokenizer,
                                                   device,
                                                   benchmark,
                                                   num_classes=num_classes, 
                                                   loss_weight=loss_weight,
                                                   goodToken=goodToken,
                                                   badToken=badToken)
        
    test_pred,test_target,test_losses=utils.eval_func(test_dataloader,
                                                      model, 
                                                      tokenizer,
                                                      device,
                                                      benchmark,
                                                      num_classes=num_classes, 
                                                      loss_weight=loss_weight,
                                                      goodToken=goodToken,
                                                      badToken=badToken)

    # best_threshold=prompting.benchmark_prompt(prompt, token_list1, token_list2, device)[1].item()

    best_threshold=utils.find_optimal_threshold(y_val.squeeze(), val_pred.squeeze(), min_recall=0.80, pos_label=False)
    y_pred=[1 if x>best_threshold else 0 for x in test_pred[:,1]]
    test_output=utils.model_evaluate(test_target.reshape(-1),y_pred)

    with open(os.path.join(output_dir,"metrics_test.txt"),'a') as f:
        f.write(f'{args.model_name},{test_output["total positive"]},{test_output["false positive"]},{test_output["false_negative"]}, \
        {test_output["precision"]},{test_output["recall"]},{test_output["f1_score"]},{test_output["AUC"]},{test_output["pr_auc"]},{best_threshold}\n')
    
    with open(os.path.join(output_dir,"y_true_pred.txt"),'w') as f:
        for x,y,z in zip(test_target.tolist(),y_pred,test_pred[:,1].tolist()):
            f.write(str(x)+","+str(y)+","+str(z)+ '\n')  
            
    # with open("y_true_pred.txt","r") as file:
    #     true_y=[]
    #     pred_y=[]
    #     prob_y=[]
    #     for line in file:
    #         x,y,z=line.strip().split(',')
    #         true_y.append(int(x))
    #         pred_y.append(int(y))
    #         prob_y.append(float(z))
    # all(x==y for x,y in zip(true_y, test_target))
    # all(x==y for x,y in zip(pred_y, y_pred))
    # all(x==y for x,y in zip(prob_y, test_pred[:,1]))
    
    print("==> performance on test set \n")
    print("")
    print("total positive: {:,} | false positive: {:,} | false_negative: {:,} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%}".\
           format(test_output["total positive"], test_output["false positive"], test_output["false_negative"], \
                 test_output["precision"], test_output["recall"], test_output["f1_score"], test_output["AUC"], test_output["pr_auc"]))


    print()
    print(f"\n===========Test Set Performance===============\n")
    print()
    y_pred=[1 if x>best_threshold else 0 for x in test_pred[:,1]]
    print(classification_report(test_target, y_pred))
    print()
    print(confusion_matrix(test_target, y_pred))  
                    
                    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prompt Learning')

    parser.add_argument("--loss_weight", action='store_true', help="weight for unbalance data")
    parser.add_argument("--train_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--val_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--test_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--train_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs position ratio in training")
    parser.add_argument("--val_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs position ratio in validation")
    parser.add_argument("--test_negative_positive_ratio",  type=int,default=20,help="Undersampling negative vs position ratio in test")
    # parser.add_argument("--test_negative_positive_ratio",  type=int,default=10,help="Undersampling negative vs position ratio in test set")
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--truncation_strategy", type=str, default="head",help="how to truncate the long length email")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                               help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--lr_scheduler_type', type=str, default="linear")
    parser.add_argument('--customized_tokenizer',  action="store_true")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")    
    parser.add_argument('--use_schedule', action="store_true")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_ratio", default=0.4, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument('--model_name', type=str, default="roberta-large")
    parser.add_argument("--output_dir", type=str, default=None, help="output folder name")
    parser.add_argument("--model_output_name",  type=str, default=None)
    parser.add_argument("--feature_name", default="preprocessed_email", type=str)
    parser.add_argument("--frozen_layers", type=int, default=0,help="freeze layers without gradient updates")
    
    args= parser.parse_args()

    if args.customized_tokenizer:
        args.output_dir="tokenizer"+"_"+args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] + "_repo"
    else:
        args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] + "_repo"

    

    seed_everything(args.seed)

    print()
    print(args)
    print()
    
#     root_dir="/opt/omniai/work/instance1/jupyter"
#     data_dir=os.path.join(root_dir,"email-complaints","datasets")

#     train_df=load_from_disk(os.path.join(data_dir,"train_df"))
#     val_df=load_from_disk(os.path.join(data_dir,"val_df"))
#     test_df=load_from_disk(os.path.join(data_dir,"test_df"))

#     train_df=train_df.filter(lambda x: x['text_length']>10)
#     val_df=val_df.filter(lambda x: x['text_length']>10)
#     test_df=test_df.filter(lambda x: x['text_length']>10)
    
#     hf_data=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})
#     hf_data.save_to_disk(os.path.join(os.getcwd(),'hf_data'))
    
    hf_data=load_from_disk(os.path.join(os.getcwd(),"hf_data"))
    
    hf_data=hf_data.select_columns(['snapshot_id', 'preprocessed_email','is_complaint'])

    def binary_label(example):
        return {"target": 1 if example["is_complaint"]=="Y" else 0}

    train_df=hf_data["train"].map(binary_label)
    val_df=hf_data["validation"].map(binary_label)
    test_df=hf_data["test"].map(binary_label)

    model_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models",args.model_name)
    config=AutoConfig.from_pretrained(model_path)
    tokenizer= AutoTokenizer.from_pretrained(model_path,model_max_length=config.max_position_embeddings-2)
    
    prefix_prompt="email review:"
    # post_prompt="The emotion of this email is " + f" {tokenizer.mask_token} ."
    post_prompt="The emotion of this email is  " + f" {tokenizer.mask_token} ."
    # prompt=[prefix_prompt,post_prompt]
    prompt=[post_prompt]
    token_list1=["positive"]
    token_list2=["negative"]
    
    wrapped_train=wrapper.prompt_wrapper(train_df, prompt, tokenizer)
    wrapped_val=wrapper.prompt_wrapper(val_df, prompt, tokenizer)
    wrapped_test=wrapper.prompt_wrapper(test_df, prompt, tokenizer)

    wrapped_text=DatasetDict({"train":wrapped_train, "validation":wrapped_val, "test":wrapped_test})

    wrapped_text.save_to_disk(os.path.join(os.getcwd(),args.model_name.split("-")[0]+'_'+'wrapped_text'))


    wrapped_text=load_from_disk(os.path.join(os.getcwd(),args.model_name.split("-")[0]+'_'+'wrapped_text'))
    
    print(wrapped_text)

#     train_df=wrapped_text["train"]
#     val_df=wrapped_text["validation"]
#     test_df=wrapped_text["test"]

#     def under_sampling_func(df,feature,negative_positive_ratio,seed=101):
#         df.set_format(type="pandas")
#         data=df[:]
#         df=utils.under_sampling(data,feature, seed, negative_positive_ratio)
#         df.reset_index(drop=True, inplace=True)  
#         df=datasets.Dataset.from_pandas(df)
#         return df
#     if args.train_undersampling:
#         train_df=under_sampling_func(wrapped_text["train"],"target",args.train_negative_positive_ratio,args.seed)
#     if args.val_undersampling:
#         val_df=under_sampling_func(wrapped_text["validation"],"target",args.val_negative_positive_ratio,args.seed)
#     if args.test_undersampling:
#         test_df=under_sampling_func(wrapped_text["test"],"target",args.test_negative_positive_ratio,args.seed)

    
#     wrapped_text=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})
    
#     print()
#     print(wrapped_text)
#     print()
    
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     main(args, wrapped_text, device, prompt, token_list1=["positive"], token_list2=["negative"])
    