# Hahow by huggingface
To train with different input string, just change the code in `raw2user` in `text_cls.py`.

## Train
```bash
bash train.sh path/to/hahow/data/ train_group.csv val_seen_group.csv
```
- `train.log` is under model path.
- Feel free to change params.

## Test
```bash
bash train.sh path/to/hahow/data/ train_group.csv test_seen_group.csv
```
- Predict file `pred.csv` under model path.
