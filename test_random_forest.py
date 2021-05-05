from mysklearn.myclassifiers import MyRandomForestClassifier

interview_table = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

interview_class_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]


def test_random_forest():
    rf = MyRandomForestClassifier()
    rf.fit(interview_table, interview_class_train, M=7, N=20, F=2)

    assert len(rf.trees) == 7

    rf = MyRandomForestClassifier()
    rf.fit(interview_table, interview_class_train, M=6, N=20, F=2)

    assert len(rf.trees) == 6