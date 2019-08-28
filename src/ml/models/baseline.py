from sklearn import svm


def get_baseline(C, gamma):
    return svm.SVC(C, 'rbf', gamma=gamma)
