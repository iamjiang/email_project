export CUDA_VISIBLE_DEVICES=0
python model_inference.py \
--model_name longformer-large-4096 \
--batch_size 2 \
--val_min_recall 0.95 \
--device cuda
