with open("./test_unseen.csv") as f:
	users = list(map(lambda x : x[:-1], f.readlines()))[1:]

with open("./hotcourse.txt") as f:
	hot_courses = list(map(lambda x: x[:-1], f.readlines()))

with open("pred_unseen_course.csv",'w') as f:
	f.write("user_id,course_id\n")
with open("pred_unseen_course.csv",'a') as f:
	for user in users:
		f.write(f'{user},{" ".join(hot_courses[:53])}\n')