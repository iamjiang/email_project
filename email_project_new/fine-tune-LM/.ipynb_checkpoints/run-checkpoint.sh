#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v2_new_email/fine-tune-LM'

export CUDA_VISIBLE_DEVICES=1
python run_language_model.py \
--model_name longformer-large-4096 \
--batch_size 4 \
--num_epochs 25 \
--es_patience 5 \
--lr 1e-5 \
--weight_decay 5e-5 \
--fp16 \
--gradient_accumulation_steps 8 \
--use_schedule \
--train_undersampling \
--val_undersampling \
--test_undersampling \
--train_negative_positive_ratio 30 \
--val_negative_positive_ratio 50 \
--test_negative_positive_ratio 100 \
--wwm_probability 0.15 &


export CUDA_VISIBLE_DEVICES=2
python run_language_model.py \
--model_name longformer-base-4096 \
--batch_size 12 \
--num_epochs 25 \
--es_patience 5 \
--lr 1e-5 \
--weight_decay 5e-5 \
--fp16 \
--gradient_accumulation_steps 2 \
--use_schedule \
--train_undersampling \
--val_undersampling \
--test_undersampling \
--train_negative_positive_ratio 30 \
--val_negative_positive_ratio 50 \
--test_negative_positive_ratio 100 \
--wwm_probability 0.15 &

export CUDA_VISIBLE_DEVICES=3
python run_language_model.py \
--model_name roberta-large \
--batch_size 8 \
--num_epochs 25 \
--es_patience 5 \
--lr 1e-5 \
--weight_decay 5e-5 \
--fp16 \
--gradient_accumulation_steps 4 \
--use_schedule \
--train_undersampling \
--val_undersampling \
--test_undersampling \
--train_negative_positive_ratio 30 \
--val_negative_positive_ratio 50 \
--test_negative_positive_ratio 100 \
--wwm_probability 0.15 &

export CUDA_VISIBLE_DEVICES=4
python run_language_model.py \
--model_name roberta-base \
--batch_size 8 \
--num_epochs 25 \
--es_patience 5 \
--lr 1e-5 \
--weight_decay 5e-5 \
--fp16 \
--gradient_accumulation_steps 4 \
--use_schedule \
--train_undersampling \
--val_undersampling \
--test_undersampling \
--train_negative_positive_ratio 30 \
--val_negative_positive_ratio 50 \
--test_negative_positive_ratio 100 \
--wwm_probability 0.15 &

export CUDA_VISIBLE_DEVICES=5
python run_language_model.py \
--model_name bigbird-roberta-large \
--batch_size 16 \
--num_epochs 25 \
--es_patience 5 \
--lr 1e-5 \
--weight_decay 5e-5 \
--fp16 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing \
--use_schedule \
--train_undersampling \
--val_undersampling \
--test_undersampling \
--train_negative_positive_ratio 30 \
--val_negative_positive_ratio 30 \
--test_negative_positive_ratio 30 \
--wwm_probability 0.15 &

# Wait for all commands to finish
wait