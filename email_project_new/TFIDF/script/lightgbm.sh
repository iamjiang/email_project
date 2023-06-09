#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v4_new_email/TFIDF'
export model_name="lightgbm"
export test_date="04_23"

python ../main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name $model_name \
--test_date $test_date \
--val_min_recall 0.9 &

python ../main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name $model_name \
--test_date $test_date \
--val_min_recall 0.91 &

python ../main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name $model_name \
--test_date $test_date \
--val_min_recall 0.92 &

python ../main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name $model_name \
--test_date $test_date \
--val_min_recall 0.93 &

python ../main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name $model_name \
--test_date $test_date \
--val_min_recall 0.94 &

python ../main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name $model_name \
--test_date $test_date \
--val_min_recall 0.95 &

python ../main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name $model_name \
--test_date $test_date \
--val_min_recall 0.96 &

python ../main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name $model_name \
--test_date $test_date \
--val_min_recall 0.97 &

python ../main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name $model_name \
--test_date $test_date \
--val_min_recall 0.98 &

python ../main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--model_name $model_name \
--test_date $test_date \
--val_min_recall 0.99 &

wait



