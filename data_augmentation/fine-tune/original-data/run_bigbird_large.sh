# export model_checkpoint=/opt/omniai/work/instance1/jupyter/transformers-models
# python tokenization_preprocess
export CUDA_VISIBLE_DEVICES=2
python main.py --model_name bigbird-roberta-large \
--train_batch_size 1 \
--test_batch_size 1 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 24 \
--fp16 \
--loss_weight \
--use_schedule \
--train_undersampling \
--train_negative_positive_ratio 5 




