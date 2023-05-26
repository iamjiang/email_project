#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v2_new_email/Fine-Tuning/short_email'

export CUDA_VISIBLE_DEVICES=6
python model_inference.py \
--model_name deberta-v2-xlarge \
--batch_size 8 \
--fp16 \
--device cuda \
--val_min_recall 0.90 &

export CUDA_VISIBLE_DEVICES=6
python model_inference.py \
--model_name deberta-v2-xlarge \
--batch_size 8 \
--fp16 \
--device cuda \
--val_min_recall 0.925 &

export CUDA_VISIBLE_DEVICES=0
python model_inference.py \
--model_name deberta-v2-xlarge \
--batch_size 8 \
--fp16 \
--device cuda \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=0
python model_inference.py \
--model_name deberta-v2-xlarge \
--batch_size 8 \
--fp16 \
--device cuda \
--val_min_recall 0.965 &

export CUDA_VISIBLE_DEVICES=7
python model_inference.py \
--model_name deberta-v2-xlarge \
--batch_size 8 \
--fp16 \
--device cuda \
--val_min_recall 0.975 &



export CUDA_VISIBLE_DEVICES=7
python model_inference.py \
--model_name deberta-v3-large \
--batch_size 8 \
--fp16 \
--device cuda \
--val_min_recall 0.90 &

export CUDA_VISIBLE_DEVICES=7
python model_inference.py \
--model_name deberta-v3-large \
--batch_size 8 \
--fp16 \
--device cuda \
--val_min_recall 0.925 &

# export CUDA_VISIBLE_DEVICES=3
# python model_inference.py \
# --model_name longformer-base-4096 \
# --batch_size 16 \
# --fp16 \
# --device cuda \
# --val_min_recall 0.950 &

# export CUDA_VISIBLE_DEVICES=4
# python model_inference.py \
# --model_name longformer-base-4096 \
# --batch_size 16 \
# --fp16 \
# --device cuda \
# --val_min_recall 0.965 &

# export CUDA_VISIBLE_DEVICES=7
# python model_inference.py \
# --model_name longformer-base-4096 \
# --batch_size 16 \
# --fp16 \
# --device cuda \
# --val_min_recall 0.975  &

# Wait for all commands to finish
wait

