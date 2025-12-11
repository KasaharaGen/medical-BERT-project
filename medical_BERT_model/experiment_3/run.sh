set -e
source ~/.bashrc
conda activate gen
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

echo 'start pretraining'


echo 'phase 2'
cd pretraining_bert_2
python pretraining_bert_2.py
cd ../

echo 'finish pretraining'
