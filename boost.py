import torch

from dectree import DecisionTree
from util import load_data


class BoostedTree:
    def __init__(self, num_models=3):
        self.num_models = num_models

    def fit(self, X, y):
        ans = torch.zeros((X.size(0), 1))
        for _ in range(self.num_models):
            tree = DecisionTree(max_depth=10)
            root = tree.fit(X, y)
            pred = tree.predict(root, X).reshape(-1, 1)
            ans += pred
            y = y - pred
        return ans


def main():
    X, y, _, _ = load_data()
    print(y)
    tree2 = DecisionTree(max_depth=10)
    root2 = tree2.fit(X, y2)
    pred2 = tree2.predict(root2, X).reshape(-1, 1)
    print(pred2)
    print(pred1 + pred2)


if __name__ == "__main__":
    main()
