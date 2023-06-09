import argparse
import pandas as pd
import numpy as np
import os

def main():

    root_dir="/opt/omniai/work/instance1/jupyter/v2_new_email/datasets/split_data/"   
        
    data_name=[x for x in os.listdir(root_dir) if x.split("_")[-2]=="pickle"]
    df=pd.DataFrame()
    for file in data_name:
        x=pd.read_pickle(os.path.join(root_dir,file))
        df=pd.concat([df,x],axis=0,ignore_index=True)
        print(file.split("_")[-1])
        
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace = True)
    
    print(df.shape)
    
if __name__=="__main__":
    
    main()
    
    