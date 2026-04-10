export CUDA_VISIBLE_DEVICES=0
model_name=Client

# 统一设置 train_epochs=10, lr=0.0005（更稳健）
EPOCHS=10
LR=0.0005
Batch_size=32

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --model 256 \
  --n_heads 8 \
  --d_ff 512 \
  --batch_size $Batch_size \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --use_L 1 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --model 256 \
  --n_heads 8 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate $LR \
  --train_epochs $EPOCHS \

  --use_L 1 \
  --itr 1
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --model 256 \
  --n_heads 8 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --use_L 1 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --model 256 \
  --n_heads 8 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --use_L 1 \
  --itr 1




