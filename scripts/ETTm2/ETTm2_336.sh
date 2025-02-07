export CUDA_VISIBLE_DEVICES=0

model_name=TimeKAN
seq_len=96
e_layers=1
down_sampling_layers=1
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=32
train_epochs=10
begin_order=1
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --train_epochs $train_epochs \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_window $down_sampling_window\
  --begin_order $begin_order