export CUDA_VISIBLE_DEVICES=3
python run_language_model.py \
--model_name roberta-large \
--batch_size 8 \
--num_epochs 10 \
--fp16 \
--gradient_accumulation_steps 4 \
--use_schedule \
--train_undersampling \
--val_undersampling \
--test_undersampling \
--train_negative_positive_ratio 5 \
--val_negative_positive_ratio 5 \
--test_negative_positive_ratio 20 

