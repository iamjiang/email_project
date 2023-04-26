export CUDA_VISIBLE_DEVICES=1
python model_inference.py \
--model_name longformer-large-4096 \
--batch_size 8 \
--val_min_recall 0.95 \
--device cuda \
--data_augmentation summarization

