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

from dataset import OneHotDataset_course_info, OneHotDataset_course_group_info
from model import MultiLabelCls_subgroup, MultiLabelCls_subgroup_2

import random
import numpy as np
import pandas as pd
import os
from pandas import DataFrame
from average_precision import mapk

TRAIN = "train_split_id_10"
DEV = "val_seen_id_split_id"
SPLITS = [TRAIN, DEV]

def main(args):
    CKPT_FOLDER = f"{args.ckpt_dir}_{args.hidden_size}_{args.batch_size}_{args.lr}_{TRAIN}_onlygroup_MultiLabelCls_subgroup"
    if not os.path.exists(CKPT_FOLDER):
        os.mkdir(CKPT_FOLDER)
    # reproduce https://pytorch.org/docs/stable/notes/randomness.html
    for module in [torch.manual_seed, random.seed, np.random.seed]:
        module(args.random_seed)

    subgroups: DataFrame = pd.read_csv("../test/data_id/course_id.csv")
    train_id: DataFrame = pd.read_csv(args.data_dir / "train_id.csv")
    idx2group: Dict[int, str] = subgroups.set_index("course_id").to_dict()["course_name"]
    group2idx: Dict[str, int] = subgroups.set_index("course_name").to_dict()["course_id"]

    del_course: Dict[str, list] = train_id.set_index("user_id").to_dict()["course_ids"]
    users: DataFrame = pd.read_csv(args.data_dir / "users.csv").fillna("").set_index("user_id")
    courses: DataFrame = pd.read_csv(args.data_dir / "courses.csv").fillna("").set_index("course_id")

    data_paths = {split: args.data_dir / f"{split}.csv" for split in SPLITS}
    
    data = {split: pd.read_csv(path).fillna("").to_dict("records") for split, path in data_paths.items()}
    datasets: Dict[str, OneHotDataset_course_info] = {
        split: OneHotDataset_course_group_info(split_data, users, courses, idx2group)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(split_ds, args.batch_size, shuffle=True,
            collate_fn=split_ds.collate_fn)
        for split, split_ds in datasets.items()
    }
    # TODO: init model and move model to target device(cpu / gpu)
    # input size is 153 (+91), maybe there is a better way to pass in model
    model = MultiLabelCls_subgroup(
        92, args.hidden_size, args.dropout, datasets[TRAIN].num_classes
    )

    if args.resume:
        ckpt = torch.load(args.ckpt_dir / args.resume)
        # load weights into model
        model.load_state_dict(ckpt)

    model = model.to(args.device)
    # TODO: init optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    # reg_loss = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    best_mapk = 0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")

    # reset training log
    with open(f"{CKPT_FOLDER}/train.log.csv", "w") as f:
        f.write("Train loss, train mapk, valid loss, valid mapk\n")

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train_loss = .0
        actual = []
        predicted = []
        model.train()
        for i, data in enumerate(dataloaders[TRAIN]):
            # move to arg.deivce
            for k in ["one_hot_user_data", "subgroup", "one_hot_course_data"]:
                data[k] = data[k].to(args.device)
            # zero the parameter grads
            optimizer.zero_grad()

            # forward, backward, optimize
            outputs = model(data)
            loss = criterion(outputs, data["subgroup"])
            loss.backward()
            optimizer.step()
            predicted += outputs.squeeze().argsort(descending=True).tolist()
            actual += data["course_ids"]
            train_loss += loss.item()
            if i % 1000 == 0:
                valid_loss =.0
                model.eval()
                actual = []
                predicted = []
                user_id = []
                with torch.no_grad():
                    for i, data in enumerate(dataloaders[DEV]):
                        for k in ["one_hot_user_data", "subgroup", "one_hot_course_data"]:
                            data[k] = data[k].to(args.device)
                        outputs = model(data)
                        predicted += outputs.squeeze().argsort(descending=True).tolist()
                        user_id += data["user_id"]
                        actual += data["course_ids"]
                        valid_loss += criterion(outputs, data["subgroup"]).item()

                predicte_exclusive = []
                for id, group in zip(user_id, predicted):
                    tmp = []
                    for course in group:
                        if id not in del_course.keys():
                            tmp.append(course)
                        elif str(course + 1) not in del_course[id].split(" "):
                            tmp.append(course)
                    predicte_exclusive.append(tmp)
                
                val_mapk = mapk(actual, predicte_exclusive, 50)
                with open(f"{CKPT_FOLDER}/train.log.csv", "a") as f:
                    f.write(
                        f"steps:{i}, ,{valid_loss :.7} ,{val_mapk:.7}\n"
                    )
                epoch_pbar.set_postfix_str(
                    f"Eval loss : {valid_loss}, mapk: {val_mapk}"
                )
                if val_mapk > best_mapk:
                    ckpt_path = f"{CKPT_FOLDER}/{val_mapk}.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    best_mapk = val_mapk 

        with open(f"{CKPT_FOLDER}/train.log.csv", "a") as f:
            f.write(f"{train_loss :.7}, {mapk(actual, predicted, 50):.7f},")

        # TODO: Evaluation loop - calculate accuracy and save model weights
        valid_loss =.0
        model.eval()
        actual = []
        predicted = []
        user_id = []
        with torch.no_grad():
            for i, data in enumerate(dataloaders[DEV]):
                for k in ["one_hot_user_data", "subgroup", "one_hot_course_data"]:
                    data[k] = data[k].to(args.device)
                outputs = model(data)
                predicted += outputs.squeeze().argsort(descending=True).tolist()
                user_id += data["user_id"]
                actual += data["course_ids"]
                valid_loss += criterion(outputs, data["subgroup"]).item()

        predicte_exclusive = []
        for id, group in zip(user_id, predicted):
            tmp = []
            for course in group:
                if id not in del_course.keys():
                    tmp.append(course)
                elif str(course + 1) not in del_course[id].split(" "):
                    tmp.append(course)
            predicte_exclusive.append(tmp)
        
        val_mapk = mapk(actual, predicte_exclusive, 50)
        with open(f"{CKPT_FOLDER}/train.log.csv", "a") as f:
            f.write(
                f" {valid_loss :.7}, {val_mapk:.7}\n"
            )
        epoch_pbar.set_postfix_str(
            f"Eval loss : {valid_loss}, mapk: {val_mapk}"
        )
        # # scheduler.step(valid_loss)

        # save best
        if val_mapk > best_mapk:
            ckpt_path = f"{CKPT_FOLDER}/{val_mapk}.pt"
            torch.save(model.state_dict(), ckpt_path)
            best_mapk = val_mapk 

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="../hahow/data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/seen_course_/",
    )

    # model
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.5)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=0)

    # data loader
    parser.add_argument("--batch_size", type=int, default=512)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument(
        "--resume", 
        type=Path,
        help="Model file under ckpt dir",
        default=None,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)