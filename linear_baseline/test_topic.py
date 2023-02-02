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

from dataset import TopicOneHot
from model import MultiLabelCls

import random
import numpy as np
import pandas as pd
from pandas import DataFrame
from average_precision import mapk


def main(args):

    subgroups: DataFrame = pd.read_csv(args.data_dir / "subgroups.csv")
    idx2group: Dict[int, str] = subgroups.set_index("subgroup_id").to_dict()["subgroup_name"]
    group2idx: Dict[str, int] = subgroups.set_index("subgroup_name").to_dict()["subgroup_id"]
    users: DataFrame = pd.read_csv(args.data_dir / "users.csv").fillna("").set_index("user_id")

    data = pd.read_csv(args.test_file).fillna("").to_dict("records")
    dataset = TopicOneHot(data, users, idx2group)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(dataset, args.batch_size, collate_fn=dataset.collate_fn)

    model = MultiLabelCls(
        153, args.hidden_size, args.dropout, dataset.num_classes
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    model = model.to(args.device)

    ids = []
    actual = []
    groups = []
    # TODO: predict dataset
    for data in dataloader:
        for k in ["one_hot_user_data", "subgroup"]:
            data[k] = data[k].to(args.device)
        outputs = model(data).argsort(descending=True).squeeze()
        ids += data["user_id"]
        groups += outputs.tolist()
        actual += data["label"]

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as f:
        f.write("user_id,subgroup\n")
        for id, group in zip(ids, groups):
            f.write(f"{id},")
            for i in range(50):
                f.write(f"{group[i] + 1}" + " \n"[i == 49])

    print(f"mapk: {mapk(actual, groups, 50)}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="../hahow/data/",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="../hahow/data/test_seen_group.csv"
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        # required=True,
        default="./ckpt/topic/32000/32000_128_1e-05.pt"
    )
    parser.add_argument("--pred_file", type=Path, default="topic_pred.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=32000)
    parser.add_argument("--dropout", type=float, default=0.5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8192)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
