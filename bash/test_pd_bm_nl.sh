for seed in {1..10}
#for seed in {1..1}
do
    for data_batchsize in 5 10 20 40 80 160 320 640 1280 2560
    do
        for net_tau in 0 0.2 0.4 0.6 0.8 1.0
        do
            sbatch ./bash/run_pd.sh python -u ./src/main.py --seed $seed --data_batchsize $data_batchsize --train_lr 0.001 --net_p 0 --net_tau $net_tau --data_len 2560 --tag bm_nl
        done
    done
done
