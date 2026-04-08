export CUDA_VISIBLE_DEVICES=0
model_name=Client

# 统一设置 train_epochs=10, lr=0.0005（更稳健）
EPOCHS=10
LR=0.0005
Batch_size=32
seq_len_input=336          # 主流设置：输入96步
pred_lengths=(96 192 336 720)
for pred_len in "${pred_lengths[@]}"; do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_${seq_len_input}_${pred_len} \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len_input \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 768 \
  --n_heads 8 \
  --d_ff 1024 \
  --batch_size $Batch_size \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --use_L 0 \
  --itr 1
done

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_${seq_len_input}_${pred_len} \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len_input \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 768 \
  --n_heads 8 \
  --d_ff 1024 \
  --batch_size $Batch_size \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --use_L 0 \
  --use_ME 0 \
  --itr 1
done

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_${seq_len_input}_${pred_len} \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len_input \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 768 \
  --n_heads 8 \
  --d_ff 1024 \
  --batch_size $Batch_size \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --use_L 0 \
  --use_ME 0 \
  --use_R 0 \
  --itr 1
done



