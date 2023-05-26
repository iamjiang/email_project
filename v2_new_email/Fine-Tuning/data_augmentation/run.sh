#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/new-email-project/data_augmentation'

# export CUDA_VISIBLE_DEVICES=1
# python back_translation.py  --device cuda  &


# export CUDA_VISIBLE_DEVICES=1
# python back_translation.py  --device cuda  &

export CUDA_VISIBLE_DEVICES=0
python masked_lm.py --model_name longformer-large-4096 --wwm_probability 0.15 --device cuda  &

export CUDA_VISIBLE_DEVICES=0
python nlpaug_data.py  \
--model_name roberta-large \
--device cuda \
--prob 0.5  &


wait
