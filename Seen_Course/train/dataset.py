from random import sample
from typing import List, Dict

from torch.utils.data import Dataset
import torch

from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import random

courses_feature = ["sub_groups"]

class OneHotDataset(Dataset):
    def __init__(
        self,
        data: List[Dict], # raw data, 
        users: DataFrame,
        label_mapping: Dict[str, int],
    ):
        self.data = data
        self.label_mapping = label_mapping
        self.mlbs = { 
            col: MultiLabelBinarizer().fit(
                users[col].apply(lambda x: x.split(","))
                ) 
            for col in users.columns
        }
        # 1 -> 0, 2 -> 1, ... ,91 -> 90
        self.mlbs["subgroup"] = MultiLabelBinarizer().fit([[i for i in range(1, 729)]])
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.users = users

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # return batch by passing collate_fn to DataLoader w/o implementing manually
        batch = {
            "user_id": [],
            "one_hot_user_data": [],
            "subgroup": [],
            "course_ids": [],
        }

        for i, s in enumerate(samples):
            batch["user_id"].append(s["user_id"])
            batch["course_ids"].append([x - 1 for x in list(map(int, str(s["course_ids"]).split(" ")))])
            if s["course_ids"]:
                subgroup_int = list(map(int, str(s["course_ids"]).split(" ")))
                # k = random.randint(1, len(subgroup_int))
            else: # avoid no label ""
                subgroup_int = []
                # k = 0
            # random part
            # subgroup_pred = random.sample(subgroup_int, k)
            # subgroup_other = [ g for g in subgroup_int if g not in subgroup_pred ]
            subgroup_pred = subgroup_int
            # subgroup_other = []
            batch["subgroup"].append(self.mlbs["subgroup"].transform([subgroup_pred]))

            for col in self.users.columns:
                # need [] before transform, it needs data to be list of lists
                one_hot = self.mlbs[col].transform([self.users.loc[s["user_id"]][col].split(",")])
                # haven't added any one hot data
                if len(batch["one_hot_user_data"]) == i:
                    batch["one_hot_user_data"].append(one_hot)
                else:
                    batch["one_hot_user_data"][i] = np.concatenate((batch["one_hot_user_data"][i], one_hot), axis=1)
            
            # add random group to features
            # batch["one_hot_user_data"][i] = np.concatenate((
            #     batch["one_hot_user_data"][i], self.mlbs["subgroup"].transform([subgroup_other]) 
            # ), axis=1)

        for k in ["one_hot_user_data", "subgroup"]:
            batch[k] = torch.tensor(np.array(batch[k]), dtype=torch.float32)
 
        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class OneHotDataset_course_info(Dataset):
    def __init__(
        self,
        data: List[Dict], # raw data, 
        users: DataFrame,
        courses: DataFrame,
        label_mapping: Dict[str, int],
    ):
        self.data = data
        self.label_mapping = label_mapping
        self.mlbs = { 
            col: MultiLabelBinarizer().fit(
                users[col].apply(lambda x: x.split(","))
                ) 
            for col in users.columns
        }
        for col in courses_feature:
            self.mlbs[col] = MultiLabelBinarizer().fit(
                courses[col].apply(lambda x: x.split(","))) 
        
        # 1 -> 0, 2 -> 1, ... ,91 -> 90
        self.mlbs["subgroup"] = MultiLabelBinarizer().fit([[i for i in range(1, 729)]])
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.users = users
        self.courses = courses

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # return batch by passing collate_fn to DataLoader w/o implementing manually
        batch = {
            "user_id": [],
            "one_hot_user_data": [],
            "subgroup": [],
            "one_hot_course_data": [],
            "course_ids": [],
        }

        for i, s in enumerate(samples):
            batch["user_id"].append(s["user_id"])
            batch["course_ids"].append([x - 1 for x in list(map(int, str(s["predict"]).split(" ")))])
            if s["predict"]:
                subgroup_int = list(map(int, str(s["predict"]).split(" ")))
                # k = random.randint(1, len(subgroup_int))
            else: # avoid no label ""
                subgroup_int = []
                # k = 0
            # random part
            # subgroup_pred = random.sample(subgroup_int, k)
            # subgroup_other = [ g for g in subgroup_int if g not in subgroup_pred ]
            subgroup_pred = subgroup_int
            # subgroup_other = []
            batch["subgroup"].append(self.mlbs["subgroup"].transform([subgroup_pred]))

            for col in self.users.columns:
                if col == "gender" or col == "recreation_names" or col == "occupation_titles":
                    continue
                # need [] before transform, it needs data to be list of lists
                one_hot = self.mlbs[col].transform([self.users.loc[s["user_id"]][col].split(",")])
                # haven't added any one hot data
                if len(batch["one_hot_user_data"]) == i:
                    batch["one_hot_user_data"].append(one_hot)
                else:
                    batch["one_hot_user_data"][i] = np.concatenate((batch["one_hot_user_data"][i], one_hot), axis=1)
            
            if s["course_ids"] == '':
                batch["one_hot_course_data"].append([[0]*92])
            else:
                if s["course_ids"][-1] == ' ':
                    s["course_ids"] = s["course_ids"][:-1]
                for course in list(map(int, str(s["course_ids"]).split(" "))):
                    for col in courses_feature:
                        one_hot = self.mlbs[col].transform([self.courses.loc[self.label_mapping[course]][col].split(",")])
                        # haven't added any one hot data
                        if len(batch["one_hot_course_data"]) == i:
                            batch["one_hot_course_data"].append(one_hot)
                        else:
                            batch["one_hot_course_data"][i] = [[x + y for x, y in zip(x, y)] for x, y in zip(batch["one_hot_course_data"][i], one_hot)]

            # add random group to features
            # batch["one_hot_user_data"][i] = np.concatenate((
            #     batch["one_hot_user_data"][i], self.mlbs["subgroup"].transform([subgroup_other]) 
            # ), axis=1)

        for k in ["one_hot_user_data", "subgroup", "one_hot_course_data"]:
            batch[k] = torch.tensor(np.array(batch[k]), dtype=torch.float32)
 
        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class OneHotDataset_course_group_info(Dataset):
    def __init__(
        self,
        data: List[Dict], # raw data, 
        users: DataFrame,
        courses: DataFrame,
        label_mapping: Dict[str, int],
    ):
        self.data = data
        self.label_mapping = label_mapping
        self.mlbs = { 
            col: MultiLabelBinarizer().fit(
                users[col].apply(lambda x: x.split(","))
                ) 
            for col in users.columns
        }
        for col in courses_feature:
            self.mlbs[col] = MultiLabelBinarizer().fit(
                courses[col].apply(lambda x: x.split(","))) 
        
        # 1 -> 0, 2 -> 1, ... ,91 -> 90
        self.mlbs["subgroup"] = MultiLabelBinarizer().fit([[i for i in range(1, 729)]])
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.users = users
        self.courses = courses

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # return batch by passing collate_fn to DataLoader w/o implementing manually
        batch = {
            "user_id": [],
            "one_hot_user_data": [],
            "subgroup": [],
            "one_hot_course_data": [],
            "course_ids": [],
        }

        for i, s in enumerate(samples):
            batch["user_id"].append(s["user_id"])
            batch["course_ids"].append([x - 1 for x in list(map(int, str(s["predict"]).split(" ")))])
            if s["predict"]:
                subgroup_int = list(map(int, str(s["predict"]).split(" ")))
                # k = random.randint(1, len(subgroup_int))
            else: # avoid no label ""
                subgroup_int = []
                # k = 0
            # random part
            # subgroup_pred = random.sample(subgroup_int, k)
            # subgroup_other = [ g for g in subgroup_int if g not in subgroup_pred ]
            subgroup_pred = subgroup_int
            # subgroup_other = []
            batch["subgroup"].append(self.mlbs["subgroup"].transform([subgroup_pred]))

            for col in self.users.columns:
                if col != "group":
                    continue
                # need [] before transform, it needs data to be list of lists
                for group in self.users.loc[s["user_id"]][col].split(","):
                    one_hot = self.mlbs[col].transform([[group]])
                # one_hot = self.mlbs[col].transform([self.users.loc[s["user_id"]][col].split(",")])
                # haven't added any one hot data
                    if len(batch["one_hot_user_data"]) == i:
                        batch["one_hot_user_data"].append(one_hot)
                    else:
                        batch["one_hot_user_data"][i] = [[x + y for x, y in zip(x, y)] for x, y in zip(batch["one_hot_user_data"][i], one_hot)]
            
            if s["course_ids"] == '':
                batch["one_hot_course_data"].append([[0]*92])
            else:
                if s["course_ids"][-1] == ' ':
                    s["course_ids"] = s["course_ids"][:-1]
                for course in list(map(int, str(s["course_ids"]).split(" "))):
                    for col in courses_feature:
                        one_hot = self.mlbs[col].transform([self.courses.loc[self.label_mapping[course]][col].split(",")])
                        # haven't added any one hot data
                        if len(batch["one_hot_course_data"]) == i:
                            batch["one_hot_course_data"].append(one_hot)
                        else:
                            batch["one_hot_course_data"][i] = [[x + y for x, y in zip(x, y)] for x, y in zip(batch["one_hot_course_data"][i], one_hot)]

            # add random group to features
            # batch["one_hot_user_data"][i] = np.concatenate((
            #     batch["one_hot_user_data"][i], self.mlbs["subgroup"].transform([subgroup_other]) 
            # ), axis=1)

        for k in ["one_hot_user_data", "subgroup", "one_hot_course_data"]:
            batch[k] = torch.tensor(np.array(batch[k]), dtype=torch.float32)
 
        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]