# export model_checkpoint=/opt/omniai/work/instance1/jupyter/transformers-models
# python tokenization_preprocess
export CUDA_VISIBLE_DEVICES=1,2,3
python main.py --model_name roberta-base \
--train_batch_size 32 \
--test_batch_size 32 \
--num_epochs 5 \
--gradient_accumulation_steps 1 \
--fp16 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5


export CUDA_VISIBLE_DEVICES=1
python main.py --model_name roberta-large \
--train_batch_size 8 \
--test_batch_size 8 \
--num_epochs 5 \
--gradient_accumulation_steps 4 \
--fp16 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5

export CUDA_VISIBLE_DEVICES=0
python main.py --model_name longformer-base-4096 \
--train_batch_size 4 \
--test_batch_size 4 \
--num_epochs 5 \
--gradient_accumulation_steps 6 \
--fp16 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5

export CUDA_VISIBLE_DEVICES=3
python main.py --model_name longformer-large-4096 \
--train_batch_size 1 \
--test_batch_size 1 \
--num_epochs 5 \
--gradient_accumulation_steps 24 \
--fp16 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5
