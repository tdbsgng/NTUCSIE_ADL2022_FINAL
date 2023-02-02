import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pickletools import optimize
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import OneHotDataset_course_info
from model import MultiLabelCls_subgroup

import random
import numpy as np
import pandas as pd
from pandas import DataFrame

def main(args):

    subgroups: DataFrame = pd.read_csv(args.data_dir / "course_id.csv")
    train_id: DataFrame = pd.read_csv(args.c_u_data / "train_id.csv")
    valid_id: DataFrame = pd.read_csv(args.c_u_data / "val_seen_id.csv")
    idx2group: Dict[int, str] = subgroups.set_index("course_id").to_dict()["course_name"]
    users: DataFrame = pd.read_csv(args.c_u_data / "users.csv").fillna("").set_index("user_id")
    courses: DataFrame = pd.read_csv(args.c_u_data / "courses.csv").fillna("").set_index("course_id")

    del_course: Dict[str, list] = train_id.set_index("user_id").to_dict()["course_ids"]
    del_val_course: Dict[str, list] = valid_id.set_index("user_id").to_dict()["course_ids"]


    data = pd.read_csv(args.test_file).fillna("").to_dict("records")
    dataset = OneHotDataset_course_info(data, users, courses, idx2group)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(dataset, args.batch_size, collate_fn=dataset.collate_fn)

    model = MultiLabelCls_subgroup(
        188, args.hidden_size, args.dropout, dataset.num_classes
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    model = model.to(args.device)

    ids = []
    actual = []
    groups = []
    user_id = []
    # TODO: predict dataset
    for data in dataloader:
        for k in ["one_hot_user_data", "subgroup", "one_hot_course_data"]:
            data[k] = data[k].to(args.device)
        outputs = model(data).argsort(descending=True).squeeze()
        ids += data["user_id"]
        user_id += data["user_id"]
        groups += outputs.tolist()
        # actual += data["course_ids"].squeeze().tolist()

    predicte_exclusive = []
    for id, group in zip(user_id, groups):
        tmp = []
        for course in group:
            if id not in del_course.keys():
                tmp.append(course)
            elif str(course + 1) not in del_course[id].split(" ") and \
                (id not in del_val_course.keys() or str(course + 1) not in del_val_course[id].split(" ")):
                tmp.append(course)
        predicte_exclusive.append(tmp)

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as f:
        f.write("user_id,course_id\n")
        for id, group in zip(ids, predicte_exclusive):
            f.write(f"{id},")
            for i in range(50):
                f.write(f"{idx2group[group[i] + 1]}" + " \n"[i == 49])



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data_id",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required= True,
        default="../hahow/data/test_seen_group.csv"
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True,
        default="./ckpt/intent/512.100.pt"
    )

    parser.add_argument(
        "--c_u_data",
        type=Path,
        help="Path to model checkpoint.",
        required=True,
        default="./ckpt/intent/512.100.pt"
    )
    parser.add_argument("--pred_file", type=Path, default="pred_seen_course.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=8192)
    parser.add_argument("--dropout", type=float, default=0.1)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
