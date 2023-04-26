import os
import re
import torch
import numpy as np
from transformers import LEDTokenizer, LEDForConditionalGeneration

import nltk
from nltk.tokenize import word_tokenize
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
nltk.data.path.append(nltk_data_dir)
from nltk.stem.snowball import SnowballStemmer

import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
# from tqdm.auto import tqdm
tqdm.pandas(position=0,leave=0)

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def main(args,df,device):
    
    root_dir="/opt/omniai/work/instance1/jupyter/transformers-models"
    model_path=os.path.join(root_dir,"LED")

    tokenizer=LEDTokenizer.from_pretrained(model_path)
    model = LEDForConditionalGeneration.from_pretrained(model_path)

    print()
    print(f"The maximal # input tokens : {tokenizer.model_max_length:,}")
    print(f"Vocabulary size : {tokenizer.vocab_size:,}")
    print(f"The # of parameters to be updated : {sum([p.nelement() for p in model.parameters() if p.requires_grad==True]):,}")
    print()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    def summarization_lm(text,model,tokenizer,device):
        input_ids=tokenizer.encode(text,max_length=tokenizer.model_max_length,truncation=True, padding="max_length",return_tensors="pt")
        input_ids=input_ids.to(device)
        model=model.to(device)
        summary_ids=model.generate(input_ids,max_length=512,num_beams=3,repetition_penalty=2.5,length_penalty=1.0,early_stopping=True)
        summary=tokenizer.decode(summary_ids.squeeze(),skip_special_tokens=True)
        return summary
    
    summarized_email=[]
    for index, row in tqdm(df.iterrows(),total=df.shape[0],leave=True, position=0):
        document=row['preprocessed_email']
        summarized_text=summarization_lm(document,model,tokenizer,device)
        summarized_email.append(summarized_text)
        
    df["summarized_email"]=summarized_email       
        
        
    def clean_re(text):
   
        text = re.sub(r"[^A-Za-z\s\d+\/\d+\/\d+\.\-,;?'\"%$]", '', text) # replace non-alphanumeric with space
        text = " ".join(word for word in text.split() if len(word)<=20) #remove long text
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\n{1,}", " ", text)
        text = re.sub(r"\t{1,}", " ", text)
        text = re.sub("_{2,}","",text)
        text = re.sub("\[\]{1,}","",text)
        text = re.sub(r"(\s\.){2,}", "",text) #convert pattern really. . . . . . .  gotcha  into really. gotcha 

        # Define regular expression pattern for address and signature
        address_pattern = r"\d+\s+[a-zA-Z0-9\s,]+\s+[a-zA-Z]+\s+\d{5}"
        signature_pattern = r"^\s*[a-zA-Z0-9\s,]+\s*$"
        # Remove address and signature from email
        text = re.sub(address_pattern, "", text)
        text = re.sub(signature_pattern, "", text)

        return text

    # ## remove all non-English Word
    # def remove_non_english(text):
    #     snow_stemmer = SnowballStemmer(language='english')
    #     words = set(nltk.corpus.words.words())
    #     text=" ".join(w for w in nltk.wordpunct_tokenize(text) if snow_stemmer.stem(w).lower() in words or not w.isalpha())
    #     return text

    # df['summarized_email'] = df['summarized_email'].progress_apply(remove_non_english)
    df['summarized_email'] = df['summarized_email'].progress_apply(clean_re)
        
    df.to_pickle(os.path.join(args.output_dir,"summarized_email_pickle"))        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='summarization to augment complaint emails')
    parser.add_argument('--output_dir', type=str, default = None)
    parser.add_argument('--device', type=str, default = "cpu")
    args= parser.parse_args()
    
    args.output_dir=os.path.join(os.getcwd(),"augmented_data")
    
    print()
    print(args)
    print()
    
    root_dir="/opt/omniai/work/instance1/jupyter/email-complaints/"
    df=pd.read_pickle(os.path.join(root_dir,"datasets","train_val_test_pickle"))
    df=df[df["data_type"]=="training_set"]
    df=df[df["is_complaint"]=="Y"]
    df['preprocessed_email'] = df['preprocessed_email'].astype(str)
    df.drop(["data_type"],inplace=True,axis=1)
    df.reset_index(drop=True, inplace=True)
    
    # output_dir=os.path.join(os.getcwd(), "dataset")
    # df=pd.read_pickle(os.path.join(output_dir,"new_dataset_pickle"))
    
    device=torch.device(args.device)
    
    main(args,df,device)