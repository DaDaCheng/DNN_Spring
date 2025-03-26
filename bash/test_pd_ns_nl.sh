for seed in {1..10}
do
    for train_noise in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
    do
        for net_tau in 0 0.2 0.4 0.6 0.8 1.0
        do
            sbatch ./bash/run_pd.sh python -u ./src/main_noise.py --seed $seed --data_batchsize 2560 --train_lr 0.001 --train_epoch 100 --net_p 0 --net_tau $net_tau --data_len 2560 --train_opt Adam --train_noise $train_noise  --savekey ns_nl
        done
    done
done
