#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v4_new_email/latency/tfidf'

python tfidf_data_prep.py \
--max_feature_num 7000 \
--num_sample 5000 \
--closed_status 
