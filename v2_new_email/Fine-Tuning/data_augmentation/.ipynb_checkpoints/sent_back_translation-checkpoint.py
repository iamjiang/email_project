import os
import re
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
nltk.data.path.append(nltk_data_dir)

import argparse
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas(position=0,leave=0)

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from transformers import MarianMTModel, MarianTokenizer
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

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
    
def format_batch_texts(language_code, batch_texts):

    formated_bach = ">>{}<< {}".format(language_code, batch_texts)

    return formated_bach

def perform_translation(batch_texts,device, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)
    
    # Generate translation using model
    inputs=tokenizer(formated_batch_texts, return_tensors="pt", padding=True)
    inputs={k:v.to(device) for k,v in inputs.items()}
    model=model.to(device)
    # translated = model.generate(**inputs,num_beams=5,early_stopping=True)
    translated = model.generate(**inputs,do_sample=True,max_length=tokenizer.model_max_length, temperature=0.75)
    
    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return translated_texts
    
def back_translate_document(document, first_tokenizer, first_model, second_tokenizer, second_model,device):
    sentences=sent_tokenize(document)
    translated_chunks=[]
    for sent in sentences:
        sent=clean_re(sent)
        if len(sent.split())>0:
            translated_texts = perform_translation(sent,device, first_model, first_tokenizer)
            back_translated_texts = perform_translation(translated_texts[0],device, second_model, second_tokenizer)
            translated_chunks.append(back_translated_texts[0].lower()+" ")
    translated_document="".join(translated_chunks)         
    return translated_document
        
        
def main(args,df,device):
    
    root_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models")
    first_model_name=args.en_2_other
    first_model_path=os.path.join(root_dir,first_model_name)
    first_tokenizer=MarianTokenizer.from_pretrained(first_model_path)
    first_model=MarianMTModel.from_pretrained(first_model_path)

    second_model_name=args.other_2_en
    second_model_path=os.path.join(root_dir,second_model_name)
    second_tokenizer=MarianTokenizer.from_pretrained(second_model_path)
    second_model=MarianMTModel.from_pretrained(second_model_path)

    print()
    print(f"The maximal # input tokens : {second_tokenizer.model_max_length:,}")
    print(f"Vocabulary size : {second_tokenizer.vocab_size:,}")
    print(f"The # of parameters to be updated : {sum([p.nelement() for p in second_model.parameters() if p.requires_grad==True]):,}")
    print()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    translated_email=[]
    for index, row in tqdm(df.iterrows(),total=df.shape[0],leave=True, position=0):
        document=row['preprocessed_email']        
        translated_text=back_translate_document(document, first_tokenizer, first_model, second_tokenizer, second_model,device)
        translated_email.append(translated_text)
        
    df["translated_email"]=translated_email   
    
    # df["translated_email"]=df['preprocessed_email'].progress_apply(back_translate_document, \
    #                                                                args=(first_tokenizer, first_model, \
    #                                                                      second_tokenizer, second_model,device))
    

    df['translated_email'] = df['translated_email'].progress_apply(clean_re)
        
    if args.deduped:
        output_dir=os.path.join(args.output_dir,'dedup')
    else:
        output_dir=os.path.join(args.output_dir,'non-dedup')
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df.to_pickle(os.path.join(output_dir,"sent_back_translation_pickle"))
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='back-translation to augment complaint emails')
    parser.add_argument('--en_2_other', type=str, default = "Helsinki-NLP-en-fr")
    parser.add_argument('--other_2_en', type=str, default = "Helsinki-NLP-fr-en")
    parser.add_argument('--output_dir', type=str, default = None)
    parser.add_argument('--deduped', action="store_true", help="keep most recent thread id or not")
    parser.add_argument('--device', type=str, default = "cpu")
    args= parser.parse_args()
    
    args.output_dir=os.path.join(os.getcwd(),"augmented_data")
    
    print()
    print(args)
    print()
    
    if args.deduped:
        data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "new-email-project","datasets","split_dedup_data")
    else:
        data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "new-email-project","datasets","split_data")
        
    data_name=[x for x in os.listdir(data_path) if x.split("_")[-2]=="pickle"]
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_pickle(os.path.join(data_path,data))
        df=pd.concat([df,x])
        # print("{:<20}{:<20,}".format(data.split("_")[-1],x.shape[0]))
    df.sort_values(by='time', inplace = True) 
    ## train: 09/2022 ~ 01/2023. validation: 02/2023  test: 03/2023
    set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1]) \
    else ("val" if (row["year"]==2023 and row["month"]==2) else "test")
    df["data_type"]=df.progress_apply(set_categories,axis=1)
    
    df=df[df["data_type"]=="train"]
    df=df[df["is_complaint"]=="Y"]
    df['preprocessed_email'] = df['preprocessed_email'].astype(str)
    df.drop(["data_type"],inplace=True,axis=1)
    df.reset_index(drop=True, inplace=True)
    
    device=torch.device(args.device)
    
    main(args,df,device)
    