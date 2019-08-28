import logging

from sacred import Experiment

from src.repository.iris import get_dataset, split_dataset
from src.ml.models.baseline import get_baseline

ex = Experiment('iris_rbf_svm')

@ex.config
def cfg():
    C = 1.0
    gamma = 0.7

@ex.automain
def run(C, gamma):
    logging.getLogger().setLevel(logging.INFO)

    dataset = get_dataset()

    (X_train, y_train), (X_test, y_test) = split_dataset(dataset)

    clf = get_baseline(C, gamma)

    clf.fit(X_train, y_train)

    logging.info(clf.score(X_test, y_test))
