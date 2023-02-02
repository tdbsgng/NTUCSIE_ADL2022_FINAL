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

from dataset import CourseOneHot
from model import MultiLabelCls

import random
import numpy as np
import pandas as pd
from pandas import DataFrame
from average_precision import mapk


def main(args):

    course_id_list: list = pd.read_csv(args.data_dir / "courses.csv")["course_id"].to_list()
    idx2course: Dict[int, str] = {}
    course2idx: Dict[str, int] = {}
    for k, v in enumerate(course_id_list):
        idx2course[k] = v
        course2idx[v] = k
    users: DataFrame = pd.read_csv(args.data_dir / "users.csv").fillna("").set_index("user_id")
    buyed = pd.read_csv(args.data_dir / "train.csv")
    buyed["course_id"] = buyed["course_id"].apply(lambda x: x.split(" ") if x else [])
    buyed = buyed.set_index("user_id").to_dict("index")

    data = pd.read_csv(args.test_file).fillna("").to_dict("records")
    dataset = CourseOneHot(data, users, course2idx)
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
    course = []
    # TODO: predict dataset
    for data in dataloader:
        for k in ["one_hot_user_data", "course"]:
            data[k] = data[k].to(args.device)
        outputs = model(data).argsort(descending=True).squeeze()
        ids += data["user_id"]
        course += outputs.tolist()
        actual += data["course"].squeeze().tolist()

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as f:
        f.write("user_id,course_id\n")
        for id, c in zip(ids, course):
            f.write(f"{id},")
            cnt = 0
            for i in range(len(c)):
                if str(args.test_file).split("/")[-1] == "test_seen.csv":
                    cid = idx2course[c[i]]
                    print(i, cnt, cid)
                    if cid not in buyed[id]["course_id"]:
                        if cnt == 0:
                            f.write(f"{cid}")
                        else:
                            f.write(f" {cid}")
                        cnt += 1
                    if cnt == 49:
                        break
                
                else:
                    f.write(f"{idx2course[c[i]]}")
                    if i < 49:
                        f.write(" ")
                    else:
                        break
            f.write("\n")


    # print(f"mapk: {mapk(actual, course, 50)}")


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
        default="../hahow/data/test_seen.csv"
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        # required=True,
        default="./ckpt/course/31500/31500_128_5e-06.pt"
    )
    parser.add_argument("--pred_file", type=Path, default="course_pred.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=31500)
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
