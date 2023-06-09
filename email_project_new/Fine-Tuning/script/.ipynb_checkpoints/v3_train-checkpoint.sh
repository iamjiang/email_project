#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v4_new_email/Fine-Tuning'

export CUDA_VISIBLE_DEVICES=0
python ../main.py \
--model_name roberta-large   \
--train_batch_size 8 \
--val_batch_size 8 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 8 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--closed_status \
--test_date 04_23 \
--train_undersampling \
--train_negative_positive_ratio 5 \
--customized_model \
--max_token_length 512 \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=1
python ../main.py \
--model_name roberta-large   \
--train_batch_size 8 \
--val_batch_size 8 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 8 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--closed_status \
--test_date 03_23 \
--train_undersampling \
--train_negative_positive_ratio 5 \
--customized_model \
--max_token_length 512 \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=2
python ../main.py \
--model_name roberta-large   \
--train_batch_size 8 \
--val_batch_size 8 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 8 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--closed_status \
--test_date 02_23 \
--train_undersampling \
--train_negative_positive_ratio 5 \
--customized_model \
--max_token_length 512 \
--val_min_recall 0.95 &


# Wait for all commands to finish
wait
