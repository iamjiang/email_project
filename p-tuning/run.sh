CUDA_DEVICES_VISIBLE=2 python main.py \
--model_checkpoint bert-base \
--model_type bert \
--prefix \
--prefix_projection \
--undersampling \
--train_negative_positive_ratio 2 \
--test_negative_positive_ratio 3 \
--train_batch_size 32 \
--test_batch_size 32 \
--num_epochs 6 \
--gradient_accumulation_steps 1 \
--fp16 \
--loss_weight \
--use_schedule 

