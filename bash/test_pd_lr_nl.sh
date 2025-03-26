for seed in {1..10}
do
    for net_tau in 0 0.2 0.4 0.6 0.8 1.0
    do
        for lr in 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5
        do
            sbatch ./bash/run_pd.sh python -u ./src/main.py --seed $seed --data_batchsize 2560 --train_lr $lr --net_p 0 --net_tau $net_tau --data_len 2560 --train_epoch 200 --tag lr_nl
        done
    done
done
