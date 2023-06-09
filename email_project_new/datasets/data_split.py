import argparse
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)

def main(data, split_num, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dfs=np.array_split(data,split_num)
    for i,f in enumerate(dfs):
        filename=f"email_data_{i}.csv"
        f.to_csv(os.path.join(output_dir,filename),index=False)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='email data preprocessing')
    parser.add_argument('--split_num', type=int, default=10)
    args= parser.parse_args()
    print()
    print(args)
    print()
    
    root_dir="/opt/omniai/work/instance1/jupyter/v2_new_email/datasets/raw_data"
    data_name=[x for x in os.listdir(root_dir) if x.split(".")[-1]=="csv"]
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_csv(os.path.join(root_dir,data))
        ### remove all missing email text
        x=x.dropna(subset=['email'])
        x=x[x.email.notna()]
        x=x[x.email.str.len()>0]
        df=pd.concat([df,x],axis=0,ignore_index=True)
        print("{:<20}{:<20,}".format(data.split("_")[2],x.shape[0]))

    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df.time.apply(lambda x: x.year)
    df['month'] = df.time.apply(lambda x: x.month)
    df['day'] = df.time.apply(lambda x: x.day)
    df.sort_values(by='time', inplace = True) 

    df=df[df.month!=8] ## 2022 August only have 4 non-complaint emails

    df['email'] = df['email'].str.replace("`", "'")
    df['email'] = df['email'].str.replace("’", "'")  

    # df= df.sample(5000)
    
    grouped_df=df.groupby('thread_id')
    sorted_groups=[group.sort_values("time",ascending=False).reset_index(drop=True) \
                   for _, group in grouped_df]
    df=pd.concat(sorted_groups).drop_duplicates(subset="thread_id", keep="first")\
    .reset_index(drop=True)

    
    split_num=args.split_num

    output_dir=os.path.join(os.getcwd(), "split_data")
    
    main(df, split_num, output_dir)