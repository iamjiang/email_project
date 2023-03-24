python centillion_download.py --model_name roberta-large_0 --model_version v1.0
python centillion_download.py --model_name roberta-base --model_version v2.0
python centillion_download.py --model_name finbert_2 --model_version v1.0
python centillion_download.py --model_name microsoft-deberta-base --model_version v1.0
python centillion_download.py --model_name allenai-longformer-base-4096 --model_version v1.0
python centillion_download.py --model_name longformer-base-4096-0 --model_version v1.0
python centillion_download.py --model_name longformer-large-4096_0 --model_version v1.0

python CTO_centillion.py --model_name finGloVe_50dim --model_version 1.0
python CTO_centillion.py --model_name finGloVe_100dim --model_version 1.0
python CTO_centillion.py --model_name finGloVe_300dim --model_version 1.0

# tar -xvzf finGloVe_100d.tar.gz  -C ./Glove_w2v/
tar -xvzf longformer-large-4096.tar.zip  -C ./longformer-large-4096/


