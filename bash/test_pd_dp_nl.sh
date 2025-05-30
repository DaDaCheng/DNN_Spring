#!/bin/bash
for seed in {1..10}
do
    for net_alpha in 0 0.2 0.4 0.6 0.8 1.0
    do
        for net_p in 0 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3
        do
            sbatch ./bash/run_pd.sh python -u ./src/main.py --seed $seed --data_batchsize 2560 --train_lr 0.001 --net_p $net_p --net_alpha $net_alpha --data_len 2560 --tag dp_nl
        done
    done
done
