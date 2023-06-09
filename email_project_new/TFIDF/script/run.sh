#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v4_new_email/TFIDF'

bash randomforest.sh


bash xgboost.sh


bash lightgbm.sh

