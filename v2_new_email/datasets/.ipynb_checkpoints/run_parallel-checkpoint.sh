#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v2_new_email/datasets'

# python data_split.py --split_num 16

N=15
for i in $(/usr/bin/seq 0 $N)
do 
    # echo $i
    python data_preprocess.py --data_index $i & 
    
done

wait

