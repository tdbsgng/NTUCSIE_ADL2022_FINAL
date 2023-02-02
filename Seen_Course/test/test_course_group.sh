python3 test_course_group.py \
    --ckpt_path "./model_ckpt/model.pt" \
    --test_file "$1/test.csv" \
    --c_u_data "$1"\
    --hidden_size 15000
    # --pred_file "/$epoch.csv