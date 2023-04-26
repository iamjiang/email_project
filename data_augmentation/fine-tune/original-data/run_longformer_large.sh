# export model_checkpoint=/opt/omniai/work/instance1/jupyter/transformers-models
# python tokenization_preprocess
# export CUDA_VISIBLE_DEVICES=0
# python main.py --model_name longformer-large-4096 \
# --train_batch_size 1 \
# --test_batch_size 1 \
# --num_epochs 10 \
# --es_patience 3 \
# --gradient_accumulation_steps 24 \
# --fp16 \
# --loss_weight \
# --use_schedule \
# --train_undersampling \
# --train_negative_positive_ratio 5 \
# --val_min_recall 0.95

export CUDA_VISIBLE_DEVICES=0
python main.py --model_name longformer-large-4096 \
--train_batch_size 1 \
--test_batch_size 1 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 24 \
--fp16 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 \
--customized_model 



