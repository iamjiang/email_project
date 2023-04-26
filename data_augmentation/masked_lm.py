import os
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM

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
    
    model_name=args.model_name
    output_path=model_name.split("-")[0]+"_"+model_name.split("-")[1]+"_"+"repo"
    model_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints/fine-tune-LM",output_path)

    config=AutoConfig.from_pretrained(model_path)
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    model=AutoModelForMaskedLM.from_pretrained(model_path)

    print()
    print(f"The maximal # input tokens : {tokenizer.model_max_length:,}")
    print(f"Vocabulary size : {tokenizer.vocab_size:,}")
    print(f"The # of parameters to be updated : {sum([p.nelement() for p in model.parameters() if p.requires_grad==True]):,}")
    print()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    
    def mask_lm(text,model,tokenizer,device):
        mask=np.random.binomial(1,args.wwm_probability,len(text.split()))
        mask_pos=np.where(mask==1)[0]
        text_list=text.split()
        for i in mask_pos:
            text_list[i]=tokenizer.mask_token

        text=" ".join(text_list)
        encoded_text=tokenizer.encode(text,add_special_tokens=False)
        encoded_text=encoded_text[0:tokenizer.model_max_length-2]
        mask_positions=[i for i,x in enumerate(encoded_text) if x==tokenizer.mask_token_id]
        input_ids=encoded_text.copy()
        input_ids=torch.tensor([input_ids])
        for mask_position in mask_positions:
            assert input_ids[0,mask_position]==tokenizer.mask_token_id
            # tokens=tokenizer.convert_ids_to_tokens(input_ids)
            # tokens_str=" ".join(tokens)
            # input_ids=torch.tensor([input_ids])
            with torch.no_grad():
                input_ids=input_ids.to(device)
                model=model.to(device)
                outputs=model(input_ids)
                predictions=outputs[0][0][mask_position].topk(5).indices.tolist()
                pred=torch.tensor([np.random.choice(predictions)])
                # print(mask_position,input_ids[mask_position], pred)
                input_ids[0,mask_position]=pred
                
        return tokenizer.decode(input_ids.squeeze())
        
    masked_email=[]
    for index, row in tqdm(df.iterrows(),total=df.shape[0],leave=True, position=0):
        document=row['preprocessed_email']
        masked_text=mask_lm(document,model,tokenizer,device)
        masked_email.append(masked_text)
        
    df["masked_email"]=masked_email   
    
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

#     df['masked_email'] = df['masked_email'].progress_apply(remove_non_english)
    df['masked_email'] = df['masked_email'].progress_apply(clean_re)
        
    df.to_pickle(os.path.join(args.output_dir,"masked_email_pickle"))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='masked_email to augment complaint emails')
    parser.add_argument('--model_name', type=str, default = "roberta-large")
    parser.add_argument('--wwm_probability', type=float, default = 0.10)
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