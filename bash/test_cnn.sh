for seed in {1..10}
do
    for net_p in 0 0.01 0.02 0.04 0.08 0.16 0.32 0.64
    do
        sbatch ./bash/run_pd.sh python -u ./src/main_cnn.py --seed $seed --data_len 2560 --train_lr 0.0001 --net_p $net_p --net_alpha 0 --train_epoch 2000 --net_channel 20 --tag cnn
    done
done
