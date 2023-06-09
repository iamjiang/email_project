#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v3_new_email/Fine-Tuning'

export CUDA_VISIBLE_DEVICES=1
python main.py --model_name longformer-base-4096 \
--train_batch_size 4 \
--val_batch_size 4 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--closed_status \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=1
python main.py --model_name longformer-base-4096 \
--train_batch_size 4 \
--val_batch_size 4 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--closed_status \
--train_undersampling \
--train_negative_positive_ratio 5 \
--customized_model \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=2
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
--closed_status \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 &

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
--closed_status \
--train_undersampling \
--train_negative_positive_ratio 5 \
--customized_model \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=4
python main.py --model_name bigbird-roberta-large   \
--train_batch_size 4 \
--val_batch_size 4 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--closed_status \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=5
python main.py --model_name bigbird-roberta-large   \
--train_batch_size 4 \
--val_batch_size 4 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--closed_status \
--train_undersampling \
--train_negative_positive_ratio 5 \
--customized_model \
--val_min_recall 0.95 &


export CUDA_VISIBLE_DEVICES=6
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
--closed_status \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=0
python main.py --model_name roberta-large \
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
--closed_status \
--train_undersampling \
--train_negative_positive_ratio 5 \
--customized_model \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=7
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
--closed_status \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=0
python main.py --model_name roberta-large \
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
--closed_status \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 &

# Wait for all commands to finish
wait
