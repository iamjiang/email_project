#!/bin/bash
export PYTHONPATH=$PYTHONPATH:'/opt/omniai/work/instance1/jupyter/new-email-project/peft/'
export PYTHONPATH=$PYTHONPATH:'/opt/omniai/work/instance1/jupyter/new-email-project/peft/src'

export CUDA_VISIBLE_DEVICES=6
python clm_peft.py \
--model_name bloom-1b1 \
--lr 1e-5 \
--peft_type P-tuning \
--num_epochs 20 \
--fp16 \
--use_schedule \
--train_batch_size 1 \
--val_batch_size 1 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--train_undersampling \
--train_negative_positive_ratio 3 \
--num_virtual_tokens 20 \
--max_length 3020 \
--text_label " neutral" " complaint" \
--deduped 

export CUDA_VISIBLE_DEVICES=6
python clm_inference.py \
--model_name bloom-1b1 \
--batch_size 1 \
--num_virtual_tokens 20 \
--peft_type P-tuning \
--text_label " neutral" " complaint" \
--deduped 


# #!/bin/bash
# export PYTHONPATH=$PYTHONPATH:'/opt/omniai/work/instance1/jupyter/new-email-project/peft/'
# export PYTHONPATH=$PYTHONPATH:'/opt/omniai/work/instance1/jupyter/new-email-project/peft/src'

# export CUDA_VISIBLE_DEVICES=1
# python clm_peft.py \
# --model_name bloom-560m \
# --lr 1e-5 \
# --peft_type P-tuning \
# --num_epochs 20 \
# --fp16 \
# --use_schedule \
# --train_batch_size 1 \
# --val_batch_size 1 \
# --gradient_accumulation_steps 16 \
# --gradient_checkpointing \
# --train_undersampling \
# --train_negative_positive_ratio 3 \
# --num_virtual_tokens 20 \
# --max_length 3820 \
# --text_label " neutral" " complaint" \
# --deduped 

# export CUDA_VISIBLE_DEVICES=1
# python clm_inference.py \
# --model_name bloom-560m \
# --batch_size 1 \
# --num_virtual_tokens 20 \
# --peft_type P-tuning \
# --text_label " neutral" " complaint" \
# --deduped 
