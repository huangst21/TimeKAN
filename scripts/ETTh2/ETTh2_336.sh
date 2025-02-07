export CUDA_VISIBLE_DEVICES=0

model_name=TimeKAN
seq_len=96
e_layers=1
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.005
d_model=32
d_ff=32
batch_size=8
begin_order=0
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size $batch_size\
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers\
  --down_sampling_window $down_sampling_window\
  --begin_order $begin_order