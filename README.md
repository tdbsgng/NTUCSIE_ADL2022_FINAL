# ADL FINAL
Final project for team 42
This folder contains other method(code) we tried.

## Requirement
```
pip3 install -r requirement.txt
```

## Download model
```
bash ./download.sh
```

## How to train

### Topic
Set the parameters in ```./linear_baseline/train_topic.sh``` first
```
cd ./linear_baseline
bash train_topic.sh
```

## Execute best 4 results on Kaggle

### Unseen group
If you want to change the test_file, you can modify the parameters in ./linear_baseline/run_unseen_topic.sh
```
cd ./linear_baseline
bash run_unseen_topic.sh
```
The output will be ```unseen_topic_pred.csv``` in the same directory.

### Seen group
If you want to change the test_file, you can modify the parameters in ./linear_baseline/run_seen_topic.sh
```
cd ./linear_baseline
bash run_seen_topic.sh
```
The output will be ```seen_topic_pred.csv``` in the same directory.

### Unseen course
If you want to change the test_file, you can modify the parameters in ./heuristic/hot_course_to_csv.py
```
cd ./heuristic
python3 hot_course_to_csv.py
```
The output will be ```pred_unseen_course.csv``` in the same directory.

### Seen course
If you want to change the test_file, you can modify the parameters in ./heuristic/heuristic.py
```
cd ./heuristic
python3 heuristic.py
```
The output will be ```pred_seen_course.csv``` in the same directory.

If there is any error please contact :
```howard89213@gmail.com```