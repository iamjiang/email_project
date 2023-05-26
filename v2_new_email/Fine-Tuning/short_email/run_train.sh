#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v2_new_email/Fine-Tuning/short_email'

export CUDA_VISIBLE_DEVICES=0
python main.py --model_name roberta-large \
--train_batch_size 32 \
--val_batch_size 32 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing \
--max_token_length 512 \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 &


export CUDA_VISIBLE_DEVICES=2
python main.py --model_name deberta-v3-large \
--train_batch_size 4 \
--val_batch_size 4 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 8 \
--max_token_length 512 \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=3
python main.py --model_name deberta-v2-xlarge \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--max_token_length 512 \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 &

wait
