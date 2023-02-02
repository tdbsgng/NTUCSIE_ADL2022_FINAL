: <<COMMENT
bash test.sh ../../hahow/data/ train_group.csv test_seen_group.csv
COMMENT

model=papluca/xlm-roberta-base-language-detection

python text_cls.py \
  --model_name_or_path ./ckpt/$task/$model \
  --data_dir $1 \
  --train_file $2 \
  --validation_file $3 \
  --max_length 256 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --weight_decay 5e-2 \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --output_dir ./ckpt/$model \
  --with_tracking \
  --seed 42 \
  --ignore_mismatched_sizes \
  # --pred \
  # --debug \
  # --max_train_steps 20