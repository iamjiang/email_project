#!/bin/bash
export PYTHONPATH=$PYTHONPATH:'/opt/omniai/work/instance1/jupyter/v2_new_email/peft'
export PYTHONPATH=$PYTHONPATH:'/opt/omniai/work/instance1/jupyter/v2_new_email/peft/src'

export CUDA_VISIBLE_DEVICES=2
python seq_inference.py \
--model_name roberta-large \
--batch_size 8 \
--num_virtual_tokens 20 \
--peft_type P-tuning \
--fp16 \
--max_token_length 512 \
--device cuda \
--val_min_recall 0.90 


export CUDA_VISIBLE_DEVICES=2
python seq_inference.py \
--model_name roberta-large \
--batch_size 8 \
--num_virtual_tokens 20 \
--peft_type P-tuning \
--fp16 \
--max_token_length 512 \
--device cuda \
--val_min_recall 0.925 

export CUDA_VISIBLE_DEVICES=2
python seq_inference.py \
--model_name roberta-large \
--batch_size 8 \
--num_virtual_tokens 20 \
--peft_type P-tuning \
--fp16 \
--max_token_length 512 \
--device cuda \
--val_min_recall 0.95

export CUDA_VISIBLE_DEVICES=2
python seq_inference.py \
--model_name roberta-large \
--batch_size 8 \
--num_virtual_tokens 20 \
--peft_type P-tuning \
--fp16 \
--max_token_length 512 \
--device cuda \
--val_min_recall 0.965


export CUDA_VISIBLE_DEVICES=2
python seq_inference.py \
--model_name roberta-large \
--batch_size 8 \
--num_virtual_tokens 20 \
--peft_type P-tuning \
--fp16 \
--max_token_length 512 \
--device cuda \
--val_min_recall 0.975
