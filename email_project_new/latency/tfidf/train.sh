#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v4_new_email/latency/tfidf'

python main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name lightgbm 

python main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name xgboost 

python main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name randomforest 




