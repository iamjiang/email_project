#!/bin/bash
export PYTHONPATH=$PYTHONPATH:'/opt/omniai/work/instance1/jupyter/v2_new_email/peft'
export PYTHONPATH=$PYTHONPATH:'/opt/omniai/work/instance1/jupyter/v2_new_email/peft/src'

export CUDA_VISIBLE_DEVICES=6
python seq_peft.py \
--model_name deberta-v3-base \
--lr 1e-5 \
--peft_type P-tuning \
--num_epochs 20 \
--fp16 \
--use_schedule \
--train_batch_size 8 \
--val_batch_size 8 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing \
--train_undersampling \
--train_negative_positive_ratio 5 \
--num_virtual_tokens 20 \
--val_min_recall 0.95 \
--max_token_length 512 &

export CUDA_VISIBLE_DEVICES=6
python seq_peft.py \
--model_name deberta-v3-large \
--lr 1e-5 \
--peft_type P-tuning \
--num_epochs 20 \
--fp16 \
--use_schedule \
--train_batch_size 8 \
--val_batch_size 8 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing \
--train_undersampling \
--train_negative_positive_ratio 5 \
--num_virtual_tokens 20 \
--val_min_recall 0.95 \
--max_token_length 512 &

# export CUDA_VISIBLE_DEVICES=6
# python clm_inference.py \
# --model_name bloom-1b1 \
# --batch_size 1 \
# --num_virtual_tokens 20 \
# --peft_type P-tuning \
# --text_label " neutral" " complaint" \
# --deduped 


export CUDA_VISIBLE_DEVICES=6
python seq_peft.py \
--model_name deberta-v2-xlarge \
--lr 1e-5 \
--peft_type P-tuning \
--num_epochs 20 \
--fp16 \
--use_schedule \
--train_batch_size 4 \
--val_batch_size 4 \
--gradient_accumulation_steps 8 \
--gradient_checkpointing \
--train_undersampling \
--train_negative_positive_ratio 5 \
--num_virtual_tokens 20 \
--val_min_recall 0.95 \
--max_token_length 512 &

# export CUDA_VISIBLE_DEVICES=1
# python clm_inference.py \
# --model_name bloom-560m \
# --batch_size 1 \
# --num_virtual_tokens 20 \
# --peft_type P-tuning \
# --text_label " neutral" " complaint" \
# --deduped 

export CUDA_VISIBLE_DEVICES=7
python seq_peft.py \
--model_name deberta-v2-xxlarge \
--lr 1e-5 \
--peft_type P-tuning \
--num_epochs 20 \
--fp16 \
--use_schedule \
--train_batch_size 2 \
--val_batch_size 2 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--train_undersampling \
--train_negative_positive_ratio 5 \
--num_virtual_tokens 20 \
--val_min_recall 0.95 \
--max_token_length 512 &

export CUDA_VISIBLE_DEVICES=7
python seq_peft.py \
--model_name roberta-large \
--lr 1e-5 \
--peft_type P-tuning \
--num_epochs 20 \
--fp16 \
--use_schedule \
--train_batch_size 8 \
--val_batch_size 8 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing \
--train_undersampling \
--train_negative_positive_ratio 5 \
--num_virtual_tokens 20 \
--val_min_recall 0.95 \
--max_token_length 512 &

wait 