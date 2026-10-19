import torch

from dectree import DecisionTree
from util import load_data


class BoostedTree:
    def __init__(self, num_models=3, learning_rate=1):
        self.num_models = num_models
        self.learning_rate = learning_rate

    def fit(self, X, y):
        ans = torch.zeros((X.size(0), 1))
        for _ in range(self.num_models):
            tree = DecisionTree(max_depth=10)
            root = tree.fit(X, y)
            pred = tree.predict(root, X).reshape(-1, 1)
            pred *= self.learning_rate
            ans += pred
            y = y - pred
        return ans


def main():
    X, y, _, _ = load_data()
    tree = BoostedTree(3)
    print(tree.fit(X, y))


if __name__ == "__main__":
    main()
