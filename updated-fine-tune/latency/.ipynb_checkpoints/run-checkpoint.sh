# export model_checkpoint=/opt/omniai/work/instance1/jupyter/transformers-models
# python tokenization_preprocess
export CUDA_VISIBLE_DEVICES=1
python run.py \
--model_name roberta-large \
--test_batch_size 64 \
--device cuda


export CUDA_VISIBLE_DEVICES=0,1,2
python run.py \
--model_name roberta-large \
--test_batch_size 64 \
--multiple_gpus \
--gpus 0 1 2

export CUDA_VISIBLE_DEVICES=0
python run.py \
--model_name longformer-base-4096 \
--test_batch_size 8 \
--device cuda

export CUDA_VISIBLE_DEVICES=0,1,2
python run.py \
--model_name longformer-base-4096 \
--test_batch_size 8 \
--multiple_gpus \
--gpus 0 1 2

export CUDA_VISIBLE_DEVICES=1
python run.py \
--model_name longformer-large-4096 \
--test_batch_size 2  \
--device cuda

export CUDA_VISIBLE_DEVICES=0,1,2
python run.py \
--model_name longformer-large-4096 \
--test_batch_size 2 \
--multiple_gpus \
--gpus 0 1 2
