from rank_bm25 import BM25Okapi
import pandas as pd
from ckip_transformers.nlp import CkipWordSegmenter
import numpy as np
from average_precision import mapk

ws_driver = CkipWordSegmenter(model="bert-base", device=0)

data_dir = "/home/wendell/ADL/final/2022-ADL-FINAL/hahow/data/"

courses = pd.read_csv(data_dir + "courses.csv")
idx2c = courses["course_id"].to_list()
c2idx = {c: i for i, c in enumerate(idx2c)}

courses_dict = courses.set_index("course_id").to_dict("index")



# ['course_id', 'course_name', 'course_price', 'teacher_id', 'teacher_intro', 'groups', 'sub_groups', 'topics', 'course_published_at_local', 'description', 'will_learn', 'required_tools', 'recommended_background', 'target_group']
columns=[
    # 'course_id',
    'course_name', 
    # 'course_price', 
    # 'teacher_id', 
    # 'teacher_intro', 
    'groups', 
    # 'sub_groups', 
    # 'topics', 
    # 'course_published_at_local', 
    'description', 
    # 'will_learn', 
    # 'required_tools',
    # 'recommended_background',
    'target_group'
]
courses['sentence'] = courses[columns].astype(str).values.sum(axis=1)
courses['sentence'] = courses["sentence"].str.replace('[^\w\s]','', regex=True)
courses['sentence'] = courses["sentence"].str.replace('[a-zA-Z _0-9]','', regex=True)
text = courses["sentence"].to_list()

courses["features"] = courses[["groups", "course_name","topics"]].astype(str).values.sum(axis=1)
features = courses["features"].to_list()

ws = ws_driver(text, batch_size=256, use_delim=False)
bm25 = BM25Okapi(ws)

test_data = pd.read_csv(data_dir + "/val_seen.csv")
test_len = len(test_data)
# test_len = 100
users = pd.read_csv(data_dir + "/users.csv")
# ['gender', 'occupation_titles', 'interests','recreation_names']
train = pd.read_csv(data_dir + "/train.csv")
train["course_id"] = train["course_id"].apply(lambda x: x.split() if x else [])
train_dict = train.set_index("user_id").to_dict("index")
test_data["buyed"] = test_data["user_id"].apply(lambda x: train_dict[x]["course_id"] if x in train_dict.keys() else "")
test_data["buyed"] = test_data["buyed"].apply(lambda x: "".join([features[c2idx[c]] for c in x]))

users_cols = ['gender', 'occupation_titles', 'interests','recreation_names'] # interest is better
users["query"] = users[["interests"]].astype(str).values.sum(axis=1)
users["query"] = users["query"].str.replace('[^\w\s]','', regex=True)
users["query"] = users["query"].str.replace('[a-zA-Z _0-9]','', regex=True)

users = users.drop(columns=users_cols)
users = users.set_index("user_id").to_dict("index")

predicted = [None] * len(test_data)
actual = test_data["course_id"].apply(lambda x: x.split(" ") if x else []).to_list()
test_data["query"] = test_data["user_id"].apply(lambda x: users[x]["query"])
test_data["query"] = test_data[["query", "buyed"]].astype(str).values.sum(axis=1)
tokenized_queries = ws_driver(test_data["query"].to_list()[:test_len], batch_size=16, use_delim=False)
for i, t in enumerate(tokenized_queries):
    score = bm25.get_scores(t)
    pred_indices = np.argsort(-score) # sort negatvie values ascending = sort orginal values descending
    predicted[i] = [idx2c[index] for index in pred_indices if index not in train_dict[test_data["user_id"][i]]["course_id"]]

print(mapk(actual[:test_len], predicted, 50))