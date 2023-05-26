#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v2_new_email/Fine-Tuning'

export CUDA_VISIBLE_DEVICES=1
python model_inference.py \
--model_name longformer-large-4096 \
--batch_size 16 \
--fp16 \
--device cuda \
--val_min_recall 0.90 &

export CUDA_VISIBLE_DEVICES=2
python model_inference.py \
--model_name longformer-large-4096 \
--batch_size 16 \
--fp16 \
--device cuda \
--val_min_recall 0.925 &

export CUDA_VISIBLE_DEVICES=4
python model_inference.py \
--model_name longformer-large-4096 \
--batch_size 16 \
--fp16 \
--device cuda \
--val_min_recall 0.950 &

export CUDA_VISIBLE_DEVICES=5
python model_inference.py \
--model_name longformer-large-4096 \
--batch_size 16 \
--fp16 \
--device cuda \
--val_min_recall 0.965 &

export CUDA_VISIBLE_DEVICES=6
python model_inference.py \
--model_name longformer-large-4096 \
--batch_size 16 \
--fp16 \
--device cuda \
--val_min_recall 0.975 &



export CUDA_VISIBLE_DEVICES=1
python model_inference.py \
--model_name longformer-large-4096 \
--batch_size 16 \
--fp16 \
--device cuda \
--customized_model \
--val_min_recall 0.90 &

export CUDA_VISIBLE_DEVICES=7
python model_inference.py \
--model_name longformer-large-4096 \
--batch_size 16 \
--fp16 \
--device cuda \
--customized_model \
--val_min_recall 0.925 &

export CUDA_VISIBLE_DEVICES=2
python model_inference.py \
--model_name longformer-large-4096 \
--batch_size 16 \
--fp16 \
--device cuda \
--customized_model \
--val_min_recall 0.950 &

export CUDA_VISIBLE_DEVICES=5
python model_inference.py \
--model_name longformer-large-4096 \
--batch_size 16 \
--fp16 \
--device cuda \
--customized_model \
--val_min_recall 0.965 &

export CUDA_VISIBLE_DEVICES=6
python model_inference.py \
--model_name longformer-large-4096 \
--batch_size 16 \
--fp16 \
--device cuda \
--customized_model \
--val_min_recall 0.975 &

# Wait for all commands to finish
wait

