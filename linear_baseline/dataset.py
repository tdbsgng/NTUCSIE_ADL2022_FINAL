from random import sample
from typing import List, Dict

from torch.utils.data import Dataset
import torch

from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import random

class TopicOneHot(Dataset):
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
        self.mlbs["subgroup"] = MultiLabelBinarizer().fit([[i for i in range(1, 92)]])
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
            "label": [],
        }

        for i, s in enumerate(samples):
            batch["user_id"].append(s["user_id"])

            if s["subgroup"]:
                subgroup_int = list(map(int, str(s["subgroup"]).split(" ")))
                # k = random.randint(1, len(subgroup_int))
            else: # avoid no label ""
                subgroup_int = []
                # k = 0
            batch["label"].append([s - 1 for s in subgroup_int])
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

class CourseOneHot(Dataset):
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
        self.mlbs["course"] = MultiLabelBinarizer().fit([ [i for i in range(728)] ])
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
            "course": [],
            "label": []
        }

        for i, s in enumerate(samples):
            batch["user_id"].append(s["user_id"])

            if s["course_id"]:
                batch["label"].append( [ self.label2idx(c) for c in s["course_id"].split(" ") ])
                # k = random.randint(1, len(subgroup_int))
            else: # avoid no label ""
                batch["label"].append([])
                # k = 0
            batch["course"].append(self.mlbs["course"].transform([batch["label"][i]]))

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

        for k in ["one_hot_user_data", "course"]:
            batch[k] = torch.tensor(np.array(batch[k]), dtype=torch.float32)
 
        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

