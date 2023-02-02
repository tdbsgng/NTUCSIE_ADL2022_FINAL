import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path
import random

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data_id",
    )
    parser.add_argument(
        "--do_shuffle",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    return args

def main(args):
    random.seed(42)
    df_course_id = pd.read_csv("./test/data_id/course_id.csv")
    df_train = pd.read_csv(args.data_dir / "train.csv")
    df_valid = pd.read_csv(args.data_dir / "val_seen.csv")
    df_test = pd.read_csv(args.data_dir / "test_seen.csv")
    df_out = pd.DataFrame(columns=["user_id", "course_ids", "predict"])
    df_train_id = pd.DataFrame(columns=["user_id", "course_ids"])
    df_val_id = pd.DataFrame(columns=["user_id", "course_ids"])
    df_val_split = pd.DataFrame(columns=["user_id", "course_ids", "predict"])
    df_train_split = pd.DataFrame(columns=["user_id", "course_ids", "predict"])
    
    course2id = df_course_id.set_index("course_name").to_dict()["course_id"]
    user2course = df_train.set_index("user_id").to_dict()["course_id"]
    user2course2 = df_valid.set_index("user_id").to_dict()["course_id"]

    df_out["user_id"] = df_test["user_id"]
    df_train_id["user_id"] = df_train["user_id"]
    df_val_id["user_id"] = df_valid["user_id"]
    df_val_split["user_id"] = df_valid["user_id"]

    courses = []
    for id in df_out["user_id"]:
        tmp = []
        for course in user2course[id].split(" "):
            tmp.append(course2id[course])
        if id in user2course2.keys():
            for course in user2course2[id].split(" "):
                tmp.append(course2id[course])
        st = ""
        for i in tmp:
            st += str(i) + " "
        courses.append(st[:-1])

    df_out["course_ids"] = courses
    df_out["predict"] = [666] * len(courses)
    df_out.to_csv(args.data_dir / "test.csv", index=False)

    courses = []
    for id in df_train_id["user_id"]:
        tmp = []
        for course in user2course[id].split(" "):
            tmp.append(course2id[course])
        st = ""
        for i in tmp:
            st += str(i) + " "
        courses.append(st[:-1])
    df_train_id["course_ids"] = courses
    df_train_id.to_csv(args.data_dir / "train_id.csv", index=False)

    courses = []
    for id in df_val_id["user_id"]:
        tmp = []
        for course in user2course2[id].split(" "):
            tmp.append(course2id[course])
        st = ""
        for i in tmp:
            st += str(i) + " "
        courses.append(st[:-1])
    df_val_id["course_ids"] = courses
    df_val_id.to_csv(args.data_dir / "val_seen_id.csv", index=False)

    val_courses = []
    pred_course = []
    for id in df_valid["user_id"]:
        st = ""
        pred = ""
        for course in user2course[id].split(" "):
            st += str(course2id[course]) + " "
        val_courses.append(st[:-1])
        for course in user2course2[id].split(" "):
            pred += str(course2id[course]) + " "
        pred_course.append(pred[:-1])
    df_val_split["course_ids"] = val_courses
    df_val_split["predict"] = pred_course
    df_val_split.to_csv(args.data_dir / "val_seen_id_split_id.csv", index=False)

    if args.do_shuffle:
        train_courses = []
        pred_courses = []
        users_ids = []
        for id in df_train["user_id"]:
            tmp = []
            for course in user2course[id].split(" "):
                tmp.append(course2id[course])
            for num in range(10):
                random.shuffle(tmp)
                st = ""
                prd = ""
                for i in tmp[:int(len(tmp)/2)]:
                    st += str(i) + " "
                for i in tmp[int(len(tmp)/2):]:
                    prd += str(i) + " "
                train_courses.append(st[:-1])
                pred_courses.append(prd[:-1])
                users_ids.append(id)
        
        df_train_split["user_id"] = users_ids
        df_train_split["predict"] = pred_courses
        df_train_split["course_ids"] = train_courses
        df_train_split.to_csv(args.data_dir / "train_split_id_10.csv", index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)