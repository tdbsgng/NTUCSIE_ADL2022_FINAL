# BM25 on hahow data
Change the `data_dir` at line 10 of `course.py` and `topic.py` to `path/to/hahow/data`.
Only test some combinations of user features and code is messy since we decided to user linear method later.

## Topic
For both validation and test, it will print mapk and produce `topic_pred.csv`, though mapk of test is meaningless.

### Validation
Change the path at line 26 to `val_seen_group.csv` or `val_unseen_group.csv`
```bash
python3 topic.py
```

### Test
Change the path at line 26 to `test_seen_group.csv` or `test_unseen_group.csv`
```bash
python3 topic.py
```

## Course
For both validation and test, it will print mapk and produce `course_pred.csv`, though mapk of test is meaningless.

### Validation
Change the path at line 47 to `val_seen.csv` or `val_unseen.csv`
```bash
python3 course.py
```

### Test
Change the path at line 47 to `test_seen.csv` or `test_unseen.csv`
```bash
python3 course.py
```