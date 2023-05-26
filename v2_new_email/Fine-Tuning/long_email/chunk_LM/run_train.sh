#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/new-email-project/Fine-Tuning'

export CUDA_VISIBLE_DEVICES=1
python main.py --model_name longformer-base-4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 12 \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 \
--customized_model \
--deduped &

export CUDA_VISIBLE_DEVICES=2
python main.py --model_name roberta-large \
--train_batch_size 32 \
--val_batch_size 32 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 \
--customized_model \
--deduped &

# export CUDA_VISIBLE_DEVICES=5
# python main.py --model_name roberta-large \
# --train_batch_size 32 \
# --val_batch_size 32 \
# --num_epochs 10 \
# --es_patience 3 \
# --gradient_accumulation_steps 1 \
# --gradient_checkpointing \
# --fp16 \
# --loss_weight \
# --use_schedule \
# --train_undersampling \
# --train_negative_positive_ratio 5 \
# --val_min_recall 0.90 \
# --deduped &

export CUDA_VISIBLE_DEVICES=3
python main.py --model_name longformer-large-4096   \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 \
--customized_model \
--deduped &

# export CUDA_VISIBLE_DEVICES=1
# python main.py --model_name longformer-large-4096   \
# --train_batch_size 2 \
# --val_batch_size 2 \
# --num_epochs 10 \
# --es_patience 3 \
# --gradient_accumulation_steps 16 \
# --gradient_checkpointing \
# --fp16 \
# --lr 1e-5 \
# --loss_weight \
# --use_schedule \
# --train_undersampling \
# --train_negative_positive_ratio 5 \
# --val_min_recall 0.90 \
# --customized_model 

# Wait for all commands to finish
wait

