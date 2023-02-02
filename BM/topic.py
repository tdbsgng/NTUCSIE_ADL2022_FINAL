from rank_bm25 import BM25Plus, BM25Okapi
import pandas as pd
from ckip_transformers.nlp import CkipWordSegmenter
import numpy as np
from average_precision import mapk

# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
ws_driver = CkipWordSegmenter(model="bert-base", device=0)

data_dir = "/home/wendell/ADL/final/2022-ADL-FINAL/hahow/data/"

courses = pd.read_csv(data_dir + "courses.csv")
idx2c = courses["course_id"].to_list()
c2idx = {c: i for i, c in enumerate(idx2c)}

courses_dict = courses.set_index("course_id").to_dict("index")

text = pd.read_csv(data_dir + "/subgroups.csv")["subgroup_name"].to_list()

courses["features"] = courses[["groups", "sub_groups"]].astype(str).values.sum(axis=1)
features = courses["features"].to_list()

ws = ws_driver(text, batch_size=256, use_delim=False)
bm25 = BM25Okapi(ws)

test_data = pd.read_csv(data_dir + "/val_unseen_group.csv")
test_len = len(test_data)
# test_len = 10
# test_data = test_data.loc[:test_len]
users = pd.read_csv(data_dir + "/users.csv")
# ['gender', 'occupation_titles', 'interests','recreation_names']
train_group = pd.read_csv(data_dir + "/train_group.csv")
train_group["subgroup"] = train_group["subgroup"].apply(lambda x: str(x).split() if x else [])

train = pd.read_csv(data_dir + "/train.csv")
train["course_id"] = train["course_id"].apply(lambda x: x.split() if x else [])
train_dict = train.set_index("user_id").to_dict("index")
test_data["buyed"] = test_data["user_id"].apply(lambda x: train_dict[x]["course_id"] if x in train_dict.keys() else "")
test_data["buyed"] = test_data["buyed"].apply(lambda x: "".join([features[c2idx[c]] for c in x]))

users_cols = ['gender', 'occupation_titles', 'interests','recreation_names'] # interest is best
users["query"] = users[["interests", "recreation_names"]].astype(str).values.sum(axis=1)
users["query"] = users["query"].str.replace('[^\w\s]','', regex=True)
users["query"] = users["query"].str.replace('[a-zA-Z _0-9]','', regex=True)

users = users.drop(columns=users_cols)
users = users.set_index("user_id").to_dict("index")

predicted = [None] * len(test_data)
actual = test_data["subgroup"].apply(lambda x: str(x).split(" ") if x else []).to_list()
test_data["query"] = test_data["user_id"].apply(lambda x: users[x]["query"])
test_data["query"] = test_data[["query", "buyed"]].astype(str).values.sum(axis=1)

tokenized_queries = ws_driver(test_data["query"].to_list()[:test_len], batch_size=256, use_delim=False)
for i, t in enumerate(tokenized_queries):
    score = bm25.get_scores(t)
    pred_indices = np.argsort(-score) # sort negatvie values ascending = sort orginal values descending
    predicted[i] = [str(index + 1) for index in pred_indices[:50]]

print(mapk(actual[:test_len], predicted, 50))

with open("./topic_pred.csv", "w") as f:
    f.write("user_id,subgroup\n")
    for i, uid in enumerate(test_data["user_id"].to_list()):
        f.write(f"{uid},")
        for id in predicted[i]:
            f.write(f"{int(id)}" + " \n"[id == predicted[i][-1]])