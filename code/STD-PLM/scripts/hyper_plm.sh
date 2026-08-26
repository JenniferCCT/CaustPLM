for dataset in PEMS03 PEMS08
do
    for rate in 0.3 0.5 0.9
    do
        python main.py \
        --data_path "../../../data/traffic/miss_data/${dataset}/true_data_SR-TR_${rate}_v1.npz" \
        --adj_filename "../../../data/traffic/${dataset}/${dataset}.csv" \
        --dataset ${dataset}MISSING \
        --desc ${dataset}_IM_SRTR_${rate}_plm \
        --sample_len 12 \
        --predict_len 12 \
        --train_ratio 0.6 \
        --val_ratio 0.2 \
        --epoch 500 \
        --val_epoch 1 \
        --test_epoch 5 \
        --batch_size 64 \
        --lr 0.001 \
        --causal 0 \
        --model transformer \
        --patience 50 \
        --sandglassAttn \
        --t_dim 64 \
        --node_emb_dim 64 \
        --node_embedding \
        --llm_layers 3 \
        --time_token \
        --dropout 0.05 \
        --trunc_k 64 \
        --weight_decay 0 \
        --task imputation \
        --trainset_dynamic_missing \
        --target_strategy 'hybrid' \
        --sag_dim 128 \
        --sag_tokens 128 \
        --input_dim 1 \
        --output_dim 1 \
        --wo_conloss \
        --node_shuffle_seed 5 
    done
done