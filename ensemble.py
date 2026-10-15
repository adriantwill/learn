from statistics import mode

import torch

from dectree import DecisionTree
from util import load_data


class Ensamble:
    def __init__(self, model, estimators, samples):
        self.model = model
        self.estimators = estimators
        self.samples = samples

    def fit(self, X, y, regression=True):
        self.ensamble = []
        for _ in range(self.estimators):
            n = torch.randint(0, len(X), (self.samples,))
            subset_X = X[n]
            subset_y = y[n]
            root = self.model.fit(subset_X, subset_y, regression=regression)
            self.ensamble.append(root)

    def predict(self, val, regresssion=True):
        prediciton = 0 if regresssion else []
        for i in range(self.estimators):
            if regresssion:
                prediciton += self.model.predict(self.ensamble[i], val)
            else:
                prediciton.append(self.model.predict(self.ensamble[i], val))
        return (prediciton / self.estimators) if regresssion else mode(prediciton)


def main():
    inputs, targets, _, _ = load_data()
    tree = DecisionTree(5)
    model = Ensamble(tree, 10, 100)
    model.fit(inputs, targets)
    model.predict(20)


if __name__ == "__main__":
    main()
