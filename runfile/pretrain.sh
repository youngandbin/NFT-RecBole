
for model in NGCFpretrain_all64
do
for dataset in bayc
do
    python 2_Pretrain.py \
        --model $model \
        --feature 'txt' \
        --dataset $dataset \
        --config 'pretrain' &

done
done