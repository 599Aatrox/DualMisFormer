export CUDA_VISIBLE_DEVICES=0

model_name=Client
seq_len_input=336          # 固定输入长度为 96
pred_lengths=(96 192 336 720)

# 安全配置（适合 12~24GB GPU）
d_model=768
d_ff=1024              # ≈ 4 * 384
batch_size=8
learning_rate=0.0005
dropout=0.1

for pred_len in "${pred_lengths[@]}"; do
    echo "Running: seq_len=${seq_len_input}, pred_len=${pred_len}"
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_${seq_len_input}_${pred_len} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len_input \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --freq h \
      --cycle_len 24 \
      --use_L 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --d_model $d_model \
      --n_heads 6 \
      --d_ff $d_ff \
      --dropout $dropout \
      --output_proj_dropout $dropout \
      --embed timeF \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs 10 \
      --itr 1 \
      --use_gpu True \
done

for pred_len in "${pred_lengths[@]}"; do
    echo "Running: seq_len=${seq_len_input}, pred_len=${pred_len}"
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_${seq_len_input}_${pred_len} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len_input \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --freq h \
      --cycle_len 24 \
      --use_L 0 \
      --use_ME 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --d_model $d_model \
      --n_heads 6 \
      --d_ff $d_ff \
      --dropout $dropout \
      --output_proj_dropout $dropout \
      --embed timeF \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs 10 \
      --itr 1 \
      --use_gpu True \
done

for pred_len in "${pred_lengths[@]}"; do
    echo "Running: seq_len=${seq_len_input}, pred_len=${pred_len}"
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_${seq_len_input}_${pred_len} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len_input \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --freq h \
      --cycle_len 24 \
      --use_L 0 \
      --use_ME 0 \
      --use_R 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --d_model $d_model \
      --n_heads 6 \
      --d_ff $d_ff \
      --dropout $dropout \
      --output_proj_dropout $dropout \
      --embed timeF \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs 10 \
      --itr 1 \
      --use_gpu True \
done