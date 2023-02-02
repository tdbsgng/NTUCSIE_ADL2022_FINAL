import pandas as pd

courses = pd.read_csv("../hahow/data/courses.csv")["course_id"].tolist()

# with open("hotcourse2.txt") as f:
#     for i in f.readlines():
#         i = i[:-1]
#         if i in courses:
#             with open("hotcourse.txt" ,'a') as g:
#                 g.write(i+'\n')

buyed_train = pd.read_csv("../hahow/data/train.csv").set_index("user_id")
buyed_train["course_id"] = buyed_train["course_id"].apply(lambda x: x.split(" ") if x else [])
buyed_train = buyed_train["course_id"].to_dict()
buyed_val_seen = pd.read_csv("../hahow/data/val_seen.csv").set_index("user_id")
buyed_val_seen["course_id"] = buyed_val_seen["course_id"].apply(lambda x: x.split(" ") if x else [])
buyed_val_seen = buyed_val_seen.to_dict()
# buyed_val_unseen = pd.read_csv("../hahow/data/val_unseen.csv").set_index("user_id")
# buyed_val_unseen["course_id"] = buyed_val_unseen["course_id"].apply(lambda x: x.split(" ") if x else [])
# buyed_val_unseen = buyed_val_unseen.to_dict()


with open("hotcourse.txt") as f:
    hot_courses = list(map(lambda x:x[:-1],f.readlines()))

with open("../hahow/data/test_seen.csv") as f:
    users = list(map(lambda x:x.split(",")[0],f.readlines()[1:]))

with open("pred_seen_course.csv",'w') as f:
    f.write("user_id,course_id\n")
with open("pred_seen_course.csv",'a') as f:
    for user in users:
        hot = []
        for hot_course in hot_courses:
            if user in buyed_val_seen:
                if hot_course not in buyed_train[user] and hot_course not in buyed_val_seen[user]:
                    hot.append(hot_course)
            else:
                if hot_course not in buyed_train[user]:
                    hot.append(hot_course)
        #print(hot)
        f.write(f'{user},{" ".join(hot)}\n')