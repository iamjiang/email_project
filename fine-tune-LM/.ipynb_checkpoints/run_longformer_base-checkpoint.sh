export CUDA_VISIBLE_DEVICES=3
python run_language_model.py \
--model_name longformer-base-4096 \
--batch_size 12 \
--num_epochs 20 \
--fp16 \
--gradient_accumulation_steps 2 \
--use_schedule \
--train_undersampling \
--val_undersampling \
--test_undersampling \
--train_negative_positive_ratio 20 \
--val_negative_positive_ratio 20 \
--test_negative_positive_ratio 20 

