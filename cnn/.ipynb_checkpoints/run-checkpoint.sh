export CUDA_VISIBLE_DEVICES=2
python train.py  \
--train_batch_size 256 \
--val_batch_size 1024 \
--lr 0.005 \
--loss_weight \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 \
--num_epoches 10 \
--es_patience 3 \
--device cuda

export CUDA_VISIBLE_DEVICES=2
python test.py  \
--batch_size 1024 \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_min_recall 0.95 \
--device cuda



