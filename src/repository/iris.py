from sklearn import datasets
from numpy.random import permutation

def get_dataset():
    iris = datasets.load_iris()
    per = permutation(iris.target.size)
    iris.data = iris.data[per]
    iris.target = iris.target[per]
    return iris


def split_dataset(dataset):
    X_train = dataset.data[:90]
    y_train = dataset.target[:90]
    X_test = dataset.data[90:]
    y_test = dataset.target[90:]
    return (X_train, y_train), (X_test, y_test)
