#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v4_new_email/latency/tfidf'

export CUDA_VISIBLE_DEVICES=0
python latency_tfidf.py \
--model_name randomforest \
--device cpu

export CUDA_VISIBLE_DEVICES=0
python latency_tfidf.py \
--model_name xgboost \
--device cpu

export CUDA_VISIBLE_DEVICES=0
python latency_tfidf.py \
--model_name lightgbm \
--device cpu

