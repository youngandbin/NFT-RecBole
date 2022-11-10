for DATASET in azuki bayc coolcats doodles meebits
do

    python 1_Baseline.py \
        --model Pop \
        --dataset $DATASET \
        --config 'baseline' &

done
