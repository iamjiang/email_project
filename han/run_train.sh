export CUDA_VISIBLE_DEVICES=3
python train.py  \
--batch_size 64 \
--num_epoches 100 \
--lr 0.05 \
--loss_weight \
--word_hidden_size 50 \
--sent_hidden_size 50 \
--es_min_delta 0.0 \
--es_patience 5 \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 \
--word_ratio 0.75 \
--sent_ratio 0.4
