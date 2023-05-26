#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v2_new_email/Fine-Tuning/long_email'

export CUDA_VISIBLE_DEVICES=7
python main.py --model_name longformer-base-4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--max_token_length 4096 \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 &


export CUDA_VISIBLE_DEVICES=6
python main.py --model_name longformer-large-4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--max_token_length 4096 \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=0
python main.py --model_name bigbird-roberta \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--max_token_length 4096 \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 \
--customized_model &

wait
