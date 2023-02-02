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

TRAIN = "train"
DEV = "val_seen"
SPLITS = [TRAIN, DEV]


def main(args):
    args.ckpt_dir = args.ckpt_dir / f"{args.hidden_size}/"
    # reproduce https://pytorch.org/docs/stable/notes/randomness.html
    for module in [torch.manual_seed, random.seed, np.random.seed]:
        module(args.random_seed)

    course_id_list: list = pd.read_csv(args.data_dir / "courses.csv")["course_id"].to_list()
    idx2course: Dict[int, str] = {}
    course2idx: Dict[str, int] = {}
    for k, v in enumerate(course_id_list):
        idx2course[k] = v
        course2idx[v] = k
    users: DataFrame = pd.read_csv(args.data_dir / "users.csv").fillna("").set_index("user_id")

    data_paths = {split: args.data_dir / f"{split}.csv" for split in SPLITS}
    
    data = {split: pd.read_csv(path).fillna("").to_dict("records") for split, path in data_paths.items()}
    datasets: Dict[str, CourseOneHot] = {
        split: CourseOneHot(split_data, users, course2idx)
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
    model = MultiLabelCls(
        153, args.hidden_size, args.dropout, datasets[TRAIN].num_classes
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

    # create ckpt dir if not exist
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    log_path = args.ckpt_dir / f"{args.hidden_size}_{args.batch_size}_{args.lr}_train.log.csv"
    with open(log_path, "w") as f:
        f.write("Train loss, train mapk, valid loss, valid mapk\n")

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train_loss = .0
        actual = []
        predicted = []
        model.train()
        for i, data in enumerate(dataloaders[TRAIN]):
            # move to arg.deivce
            for k in ["one_hot_user_data", "course"]:
                data[k] = data[k].to(args.device)
            # zero the parameter grads
            optimizer.zero_grad()

            # forward, backward, optimize
            outputs = model(data)
            loss = criterion(outputs, data["course"])
            loss.backward()
            optimizer.step()
            predicted += [ p[:50] for p in outputs.squeeze().argsort(descending=True).tolist() ]
            actual += data["label"]

            train_loss += loss.item()

        train_loss /= len(datasets[TRAIN])


        with open(log_path, "a") as f:
            f.write(f"{train_loss :.7}, {mapk(actual, predicted, 50):.7f},")

        # TODO: Evaluation loop - calculate accuracy and save model weights
        valid_loss =.0
        model.eval()
        actual = []
        predicted = []
        with torch.no_grad():
            for i, data in enumerate(dataloaders[DEV]):
                for k in ["one_hot_user_data", "course"]:
                    data[k] = data[k].to(args.device)
                outputs = model(data)
                predicted += [ p[:50] for p in outputs.squeeze().argsort(descending=True).tolist() ]
                actual += data["label"]
                valid_loss += criterion(outputs, data["course"]).item()
        
        valid_loss /= len(datasets[DEV])
        val_mapk = mapk(actual, predicted, 50)
        with open(log_path, "a") as f:
            f.write(
                f" {valid_loss :.7}, {val_mapk:.7}\n"
            )
        epoch_pbar.set_postfix_str(
            f"Eval loss : {valid_loss}, mapk: {val_mapk}"
        )
        # # scheduler.step(valid_loss)

        # save best
        if val_mapk > best_mapk:
            ckpt_path = f"{args.ckpt_dir}/{args.hidden_size}_{args.batch_size}_{args.lr}.pt"
            torch.save(model.state_dict(), ckpt_path)
            best_mapk = val_mapk 
        with open(log_path, "a") as f:
            f.write(
                f"best mapk {best_mapk}\n"
            )
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
        # default="./ckpt/topic/",
        default="./ckpt/course/",
    )

    # model
    parser.add_argument("--random_seed", type=int, default=4242)
    parser.add_argument("--hidden_size", type=int, default=60000)
    parser.add_argument("--dropout", type=float, default=0.5)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=0)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
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
