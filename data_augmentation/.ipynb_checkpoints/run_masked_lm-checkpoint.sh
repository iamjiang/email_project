export CUDA_VISIBLE_DEVICES=2
python masked_lm.py  \
--model_name  longformer-large-4096 \
--wwm_probability 0.15 \
--device cuda 
