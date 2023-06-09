#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v3_new_email/Fine-Tuning'
export model_name="longformer-large-4096"
export code="model_inference.py"
export batch_size=4

export CUDA_VISIBLE_DEVICES=0
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--customized_model \
--model_max_length 4096 \
--val_min_recall 0.90 &

export CUDA_VISIBLE_DEVICES=0
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--customized_model \
--model_max_length 4096 \
--val_min_recall 0.91 &

export CUDA_VISIBLE_DEVICES=0
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--customized_model \
--model_max_length 4096 \
--val_min_recall 0.92 &

export CUDA_VISIBLE_DEVICES=1
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--customized_model \
--model_max_length 4096 \
--val_min_recall 0.93 &

export CUDA_VISIBLE_DEVICES=1
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--customized_model \
--model_max_length 4096 \
--val_min_recall 0.94 &

export CUDA_VISIBLE_DEVICES=1
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--customized_model \
--model_max_length 4096 \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=2
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--customized_model \
--model_max_length 4096 \
--val_min_recall 0.96 &

export CUDA_VISIBLE_DEVICES=2
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--customized_model \
--model_max_length 4096 \
--val_min_recall 0.97 &

export CUDA_VISIBLE_DEVICES=2
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--customized_model \
--model_max_length 4096 \
--val_min_recall 0.98 &

export CUDA_VISIBLE_DEVICES=3
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--customized_model \
--model_max_length 4096 \
--val_min_recall 0.99 &

export CUDA_VISIBLE_DEVICES=3
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--val_min_recall 0.90 &

export CUDA_VISIBLE_DEVICES=3
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--val_min_recall 0.91 &

export CUDA_VISIBLE_DEVICES=4
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--val_min_recall 0.92 &

export CUDA_VISIBLE_DEVICES=4
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--val_min_recall 0.93 &

export CUDA_VISIBLE_DEVICES=4
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--val_min_recall 0.94 &

export CUDA_VISIBLE_DEVICES=5
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=5
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--val_min_recall 0.96 &

export CUDA_VISIBLE_DEVICES=5
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--val_min_recall 0.97 &

export CUDA_VISIBLE_DEVICES=6
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--val_min_recall 0.98 &

export CUDA_VISIBLE_DEVICES=6
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--val_min_recall 0.99 &

export CUDA_VISIBLE_DEVICES=7
python $code \
--model_name bigbird-roberta-large \
--batch_size $batch_size \
--fp16 \
--device cuda \
--customized_model \
--model_max_length 4096 \
--val_min_recall 0.90 &


# Wait for all commands to finish
wait

