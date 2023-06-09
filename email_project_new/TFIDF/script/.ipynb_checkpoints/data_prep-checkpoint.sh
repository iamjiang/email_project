#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v4_new_email/TFIDF'

python ../tfidf_data_prep.py \
--max_feature_num 7000 \
--validation_split 0.2 \
--closed_status \
--test_date 04_23 &

python ../tfidf_data_prep.py \
--max_feature_num 7000 \
--validation_split 0.2 \
--closed_status \
--test_date 03_23 &

python ../tfidf_data_prep.py \
--max_feature_num 7000 \
--validation_split 0.2 \
--closed_status \
--test_date 02_23 &

wait


