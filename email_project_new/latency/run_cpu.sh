#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v4_new_email/latency'

export CUDA_VISIBLE_DEVICES=0
python latency.py \
--model_name roberta-large \
--batch_size 128  \
--fp16 \
--customized_model \
--model_max_length 512 \
--closed_status \
--num_sample 5000 \
--device cpu \
--file_name latency_throughput_cpu.txt

export CUDA_VISIBLE_DEVICES=0
python latency.py \
--model_name deberta-v3-large \
--batch_size 32  \
--fp16 \
--model_max_length 512 \
--closed_status \
--num_sample 5000 \
--device cpu \
--file_name latency_throughput_cpu.txt

export CUDA_VISIBLE_DEVICES=0
python latency.py \
--model_name longformer-base-4096 \
--batch_size 8  \
--fp16 \
--customized_model \
--model_max_length 4096 \
--closed_status \
--num_sample 5000 \
--device cpu \
--file_name latency_throughput_cpu.txt

export CUDA_VISIBLE_DEVICES=0
python latency.py \
--model_name longformer-large-4096 \
--batch_size 8  \
--fp16 \
--customized_model \
--model_max_length 4096 \
--closed_status \
--num_sample 5000 \
--device cpu \
--file_name latency_throughput_cpu.txt

export CUDA_VISIBLE_DEVICES=0
python latency.py \
--model_name bigbird-roberta-large \
--batch_size 4  \
--fp16 \
--customized_model \
--model_max_length 4096 \
--closed_status \
--num_sample 5000 \
--device cpu \
--file_name latency_throughput_cpu.txt

