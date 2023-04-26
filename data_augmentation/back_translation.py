import os
import re
import nltk
from nltk.tokenize import word_tokenize
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
nltk.data.path.append(nltk_data_dir)
from nltk.stem.snowball import SnowballStemmer

import argparse
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas(position=0,leave=0)
import textwrap
wrapper = textwrap.TextWrapper(width=150) 
from IPython.display import HTML
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from transformers import MarianMTModel, MarianTokenizer
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

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
    translated = model.generate(**inputs,num_beams=5,early_stopping=True)
    # translated = model.generate(**inputs,do_sample=True,max_length=tokenizer.model_max_length, temperature=0.8)
    
    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return translated_texts
    
def back_translate_document(document, first_tokenizer, first_model, second_tokenizer, second_model,device):
    # Encode the entire document to get the number of subword tokens
    input_ids=first_tokenizer.encode(document,return_tensors="pt")
    num_subword_tokens=input_ids.shape[1]
    
    # Split the document into smaller chunks of maximum length 512 subword tokens
    max_subword_length=first_tokenizer.model_max_length-2
    
#     num_chunks=(num_subword_tokens+max_subword_length-1)//max_subword_length
#     num_full_chunks=num_subword_tokens//max_subword_length

#     chunk_lengths=[max_subword_length]*num_full_chunks
#     if num_subword_tokens%max_subword_length!=0:
#         chunk_lengths.append(num_subword_tokens%max_subword_length)
    
    #subtracting 1 from num_subword_tokens and add 1 afterward is to ensure the we round up to the nearest integer when computing the # of chunks
    num_chunks=(num_subword_tokens-1)//max_subword_length+1 
    chunk_lengths=[num_subword_tokens//num_chunks]*num_chunks
    #In case the division may not result in exact integer quotient, the leftover subword tokens needed to be distributed among the first few chunks
    for i in range(num_subword_tokens%num_chunks):
        chunk_lengths[i]+=1
        
    translated_chunks=[]
    start=0
    for length in chunk_lengths:
        end=start+length
        chunk_texts=first_tokenizer.decode(input_ids[0,start:end])
        
        translated_texts = perform_translation(chunk_texts,device, first_model, first_tokenizer)
        
        back_translated_texts = perform_translation(translated_texts[0],device, second_model, second_tokenizer)
        
        translated_chunks.append(back_translated_texts[0].lower()+" ")
        
        start=end
        
    #join the translated chunks into a single document and return it
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

#     ## remove all non-English Word
#     def remove_non_english(text):
#         snow_stemmer = SnowballStemmer(language='english')
#         words = set(nltk.corpus.words.words())
#         text=" ".join(w for w in nltk.wordpunct_tokenize(text) if snow_stemmer.stem(w).lower() in words or not w.isalpha())
#         return text

#     df['translated_email'] = df['translated_email'].progress_apply(remove_non_english)
    df['translated_email'] = df['translated_email'].progress_apply(clean_re)
        
    df.to_pickle(os.path.join(args.output_dir,"back_translation_pickle"))
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='back-translation to augment complaint emails')
    parser.add_argument('--en_2_other', type=str, default = "Helsinki-NLP-en-fr")
    parser.add_argument('--other_2_en', type=str, default = "Helsinki-NLP-fr-en")
    parser.add_argument('--output_dir', type=str, default = None)
    parser.add_argument('--device', type=str, default = "cpu")
    args= parser.parse_args()
    
    args.output_dir=os.path.join(os.getcwd(),"augmented_data")
    
    print()
    print(args)
    print()
    
    root_dir="/opt/omniai/work/instance1/jupyter/email-complaints/"
    df=pd.read_pickle(os.path.join(root_dir,"datasets","new_train_val_test_pickle"))
    df=df[df["data_type"]=="training_set"]
    df=df[df["is_complaint"]=="Y"]
    df['preprocessed_email'] = df['preprocessed_email'].astype(str)
    df.drop(["data_type"],inplace=True,axis=1)
    df.reset_index(drop=True, inplace=True)
    
    # output_dir=os.path.join(os.getcwd(), "dataset")
    # df=pd.read_pickle(os.path.join(output_dir,"new_dataset_pickle"))
    
    device=torch.device(args.device)
    
    main(args,df,device)
    