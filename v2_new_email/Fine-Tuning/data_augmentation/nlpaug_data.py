import os
import re
import random
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas(position=0,leave=0)

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
nltk.data.path.append(nltk_data_dir)
stopwords_file = open(nltk_data_dir + '/corpora/stopwords/english')
stopwords_list = stopwords_file.readlines()
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet 

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

import logging
logging.getLogger().setLevel(logging.ERROR)

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '',',','.']

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(text, n):
    words=word_tokenize(text)
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords_list]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return " ".join(new_words)

class WordAugmentor():
    def __init__(self,seed,prob):
        self.seed=seed
        self.prob=prob
        
    def augment(self,corpus,aug):
        sentences=sent_tokenize(corpus)
        for idx,sent in enumerate(sentences):
            randNum = random.random()
            if randNum <= self.prob:
                augmented_text=aug.augment(sent,n=1)
                augmented_text = " ".join([re.sub(r'<unk>',"",token) for token in augmented_text.split() ])
                sentences[idx] = augmented_text
                
        return ".".join(sentences)
        

def main(args,df,device):
    model_name=args.model_name.split("-")[0]+"_"+args.model_name.split("-")[1]+"_"+"repo"
    root_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "v2_new_email/fine-tune-LM")
    model_path=os.path.join(root_dir,model_name)

    aug_substitute = naw.ContextualWordEmbsAug(model_path=model_path, aug_p=0.2,  \
                                               action="substitute",stopwords=stopwords_list)
    
    aug_insert = naw.ContextualWordEmbsAug(model_path=model_path, aug_p=0.1,  \
                                           action="insert", top_k=20,stopwords=stopwords_list)
    
    aug = nafc.Sequential([
        aug_substitute,
        aug_insert
    ])
    
    aug_email=[]
    augmentor = WordAugmentor(args.seed, args.prob)
    for index, row in tqdm(df.iterrows(),total=df.shape[0],leave=True, position=0):
        document=row['preprocessed_email']
        aug_text=augmentor.augment(document,aug)
        aug_email.append(aug_text)
        
    df["aug_email"]=aug_email 
    
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

    # df['aug_email'] = df['aug_email'].progress_apply(remove_non_english)
    df['aug_email'] = df['aug_email'].progress_apply(clean_re)
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    df.to_pickle(os.path.join(output_dir,"aug_email_pickle"))        
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='nlp_aug to augment complaint emails')
    parser.add_argument('--seed', type=int, default = 101)
    parser.add_argument('--prob', type=float, default = 0.5)
    parser.add_argument('--output_dir', type=str, default = None)
    parser.add_argument('--model_name', type=str, required = True)
    parser.add_argument('--device', type=str, default = "cpu")
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
