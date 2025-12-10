export CUDA_VISIBLE_DEVICES=0

model_name=Client
seq_len_input=336          # 主流设置：输入96步
pred_lengths=(96 192 336 720)

# 安全配置（Traffic 862维，高维！）
d_model=384
d_ff=1536                 # 4 * 384
batch_size=4              # ⚠️ 关键！不能大
learning_rate=0.001
weight_decay=3e-4
dropout=0.2

for pred_len in "${pred_lengths[@]}"; do
    echo "Running Traffic: seq_len=${seq_len_input}, pred_len=${pred_len}"
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id traffic_${seq_len_input}_${pred_len} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len_input \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --freq h \                  # Traffic 是小时级
      --cycle_len 24 \            # 日周期
      --use_L 1 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --d_model $d_model \
      --n_heads 6 \               # 384 % 6 == 0
      --d_ff $d_ff \
      --dropout $dropout \
      --output_proj_dropout $dropout \
      --batch_size $batch_size \  # ⚠️ 必须指定！
      --learning_rate $learning_rate \
      --train_epochs 20 \
      --patience 3 \
      --itr 1 \
      --use_gpu True \
      --use_amp                   # 启用混合精度
done