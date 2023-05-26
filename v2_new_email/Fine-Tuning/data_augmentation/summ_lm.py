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

        text = re.sub(r"[^A-Za-z\s\d+\/\d+\/\d+\.\:\-,;?'\"%$]", '', text) # replace non-alphanumeric with space
        #remove long text such as encrypted string
        text = " ".join(word for word in text.split() if len(word)<=20) 

        #remove non-alphanumerc characters: '\u200c\xa0\u200c\xa0\u200c\xa0\n'
        text = re.sub(r"\u200c\xa0+",'',text)

        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\n{1,}", " ", text)
        text = re.sub(r"\t{1,}", " ", text)
        text = re.sub("_{2,}","",text)
        text = re.sub("\[\]{1,}","",text)
        text = re.sub(r"(\s\.){2,}", "",text) #convert pattern really. . . . . . .  gotcha  into really. gotcha 

        # Define regular expression pattern for address and signature
        address_pattern = r"\d+\s+[a-zA-Z0-9\s,]+\s+[a-zA-Z]+\s+\d{5}"
        # signature_pattern = r"^\s*[a-zA-Z0-9\s,]+\s*$"

        url_pattern = \
        "(?:https?:\\/\\/)?(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"

        us_phone_num_pattern = \
        '\(?\d{3}\)?[.\-\s]?\d{3}[.\-\s]?\d{4}'

        email_id_pattern = \
        "([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        # Remove address and signature from email
        text = re.sub(address_pattern, "", text)
        # text = re.sub(signature_pattern, "", text)
        text = re.sub(url_pattern, "", text)
        text = re.sub(us_phone_num_pattern, "", text)
        text = re.sub(email_id_pattern, "", text)

        return text

    # ## remove all non-English Word
    # def remove_non_english(text):
    #     snow_stemmer = SnowballStemmer(language='english')
    #     words = set(nltk.corpus.words.words())
    #     text=" ".join(w for w in nltk.wordpunct_tokenize(text) if snow_stemmer.stem(w).lower() in words or not w.isalpha())
    #     return text

    # df['summarized_email'] = df['summarized_email'].progress_apply(remove_non_english)
    df['summarized_email'] = df['summarized_email'].progress_apply(clean_re)
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    df.to_pickle(os.path.join(args.output_dir,"summarized_email_pickle"))        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='summarization to augment complaint emails')
    parser.add_argument('--output_dir', type=str, default = None)
    parser.add_argument('--device', type=str, default = "cuda")
    args= parser.parse_args()
    
    args.output_dir=os.path.join(os.getcwd(),"augmented_data")
    
    print()
    print(args)
    print()
    
    data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "v2_new_email","datasets","split_data")
        
    data_name=[x for x in os.listdir(data_path) if x.split("_")[-2]=="pickle"]
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_pickle(os.path.join(data_path,data))
        df=pd.concat([df,x],axis=0,ignore_index=True)
        # print("{:<20}{:<20,}".format(data.split("_")[-1],x.shape[0]))
    
    ### only keep emails with status=closed
    df=df[df.state=="closed"]
    
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace = True) 
    ## train: 09/2022 ~ 01/2023. validation: 02/2023  test: 03/2023
    set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1,2]) \
    else ("val" if (row["year"]==2023 and row["month"]==3) else "test")
    df["data_type"]=df.progress_apply(set_categories,axis=1)
    
    df=df[(df["data_type"]=="train") | (df["data_type"]=="val")]
    df=df[df["is_complaint"]=="Y"]
    df['preprocessed_email'] = df['preprocessed_email'].astype(str)
    df.reset_index(drop=True, inplace=True)
    
    device=torch.device(args.device)
    
    main(args,df,device)