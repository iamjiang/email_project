import argparse
import time
import datetime
import os
from tqdm.auto import tqdm
import numpy as np
from datasets import load_dataset, load_metric, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer,AutoConfig
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(f"The number of GPUs is {torch.cuda.device_count()}")

# %%
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def format_time(elapsed):
    #### Takes a time in seconds and returns a string hh:mm:ss
    elapsed_rounded=int(round(elapsed)) ### round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded))

def main(args,df):
    
    model_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models",args.model_name)
    config=AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path,model_max_length=config.max_position_embeddings-2)
    vocab_size=tokenizer.vocab_size
    def batch_iterator():
        for i in range(0,len(df),args.batch_size):
            yield df[i:i+args.batch_size]["preprocessed_email"]
            
    t1=time.time()
    new_tokenizer=tokenizer.train_new_from_iterator(batch_iterator(),vocab_size=vocab_size)
    print(f"It took {format_time(time.time()-t1)} to train new tokenizer")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    new_tokenizer.save_pretrained(os.path.join(args.output_dir,"JPMC-email-tokenizer"))
        
#     print("")
#     print(bcolors.OKBLUE+"Before training BPE tokenizer"+bcolors.ENDC)
#     print("")
#     for i in range(5):
#         print(len(tokenizer(df[i]['preprocessed_email'])['input_ids']))

#     # sample=transcript.shuffle(seed=42).select(range(10000))

#     print("")
#     print(bcolors.OKBLUE+"After training our own BPE tokenizer"+bcolors.ENDC)
#     print("")
#     for i in range(5):
#         print(len(new_tokenizer(df[i]['preprocessed_email'])['input_ids']))

    new_tokenizer=tokenizer.from_pretrained(os.path.join(args.output_dir,"JPMC-email-tokenizer"))

    # %%
    print("{:<40}{:<20,}".format("Vocabulary size in pretrained model",tokenizer.vocab_size))
    print("{:<40}{:<20,}".format("Vocabulary size in customized model",new_tokenizer.vocab_size))

    # %%
    print()
    print("*"*50)
    print()
    print(bcolors.OKBLUE+"Most rare tokens in the vocaburary of existing pretrained tokenizer: \n"+bcolors.ENDC)
    tokens=sorted(tokenizer.vocab.items(), key=lambda x: x[1],reverse=True)
    for token, index in tokens[:12]:
        print("{:<25}{:<15}".format(tokenizer.convert_tokens_to_string(token),index))

    print()
    print("*"*50)
    print()
    print(bcolors.OKBLUE+"Most rare tokens in the vocaburary of our own customized tokenizer: \n"+bcolors.ENDC)
    tokens=sorted(new_tokenizer.vocab.items(), key=lambda x: x[1],reverse=True)
    for token, index in tokens[:12]:
        print("{:<25}{:<15}".format(tokenizer.convert_tokens_to_string(token),index))

    print()
    print("*"*50)
    print()

    # %%
    print()
    print("*"*50)
    print()
    print(bcolors.OKBLUE+"Most frequent tokens in the vocaburary of existing pretrained tokenizer: \n"+bcolors.ENDC)
    tokens=sorted(tokenizer.vocab.items(), key=lambda x: x[1],reverse=False)
    for token, index in tokens[257:270]:
        print("{:<25}{:<15}".format(tokenizer.convert_tokens_to_string(token),index))

    print()
    print("*"*50)
    print()
    print(bcolors.OKBLUE+"Most frequent tokens in the vocaburary of our own customized tokenizer: \n"+bcolors.ENDC)
    tokens=sorted(new_tokenizer.vocab.items(), key=lambda x: x[1],reverse=False)
    for token, index in tokens[257:270]:
        print("{:<25}{:<15}".format(tokenizer.convert_tokens_to_string(token),index))

    print()
    print("*"*50)
    print()

    
if __name__=="__main__":
    
    argparser = argparse.ArgumentParser("Tokenization")
    argparser.add_argument('--batch_size', type=int, default=1000)
    argparser.add_argument("--model_name", default="roberta-large", type=str, help="pretrained model name")
    argparser.add_argument("--output_dir", default="roberta-large-finetuned", type=str, help="output folder name")

    args = argparser.parse_args()
    args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] + "_repo"
    
    print(args)
    
    data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints","few-shot")
    hf_data=load_from_disk(os.path.join(data_path,"hf_data"))
    hf_data=hf_data.select_columns(['snapshot_id', 'preprocessed_email','is_complaint'])

    print(hf_data)
    
    # train_data=hf_data["train"]
    
    df_all=concatenate_datasets([hf_data["train"], \
                                 hf_data["validation"],\
                                 hf_data["test"]
                                ])
    
    main(args,df_all)
    # main(args,train_data)
    
    
## python train_tokenizer.py --model_name longformer-large-4096