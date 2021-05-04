import mysklearn.myclassifiers
from mysklearn.myclassifiers import MyRandomForestClassifier


import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 

iphone_x = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
iphone_y = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]


def test_random_forest():
    rf = MyRandomForestClassifier()
    rf.fit(iphone_x, iphone_y, M=7, N=20, F=2)

    assert len(rf.trees) == 7

    rf = MyRandomForestClassifier()
    rf.fit(iphone_x, iphone_y, M=6, N=20, F=2)

    assert len(rf.trees) == 6