import math

import torch

from util import load_data


def mse(y_hat, y):
    summation = ((y - y_hat) ** 2).sum()
    n = y.shape[0]
    return summation / n


def gini(y):
    n = y.shape[0]
    _, counts = torch.unique(y, return_counts=True)
    purity = ((counts / n) ** 2).sum()
    return 1 - purity


class DecisionTreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth

    def fit(self, X, y, depth=0, regression=True):
        exception_node = DecisionTreeNode(
            y.mean() if regression else y.mode(dim=0).values.item()
        )
        if depth >= self.max_depth or torch.all(y == y[0]):
            return exception_node
        if depth == 0:
            index = X[:, 0].argsort()
            X = X[index]
            y = y[index]
        best_fit = math.inf
        best_split = -1
        for i in range(1, len(X)):
            if X[i - 1, 0] == X[i, 0]:
                continue
            m = y.shape[0]
            fit_variables = []
            for j in range(2):
                left_y_split = y[i:] if j else y[:i]
                fit_variables.append(
                    mse(left_y_split.mean(), left_y_split)
                    if regression
                    else gini(left_y_split)
                )
                fit_variables.append(left_y_split.shape[0])
            left_err, left_m, right_err, right_m = fit_variables
            fit = left_m / m * left_err + right_m / m * right_err
            if fit < best_fit:
                best_fit = fit
                best_split = i
        if best_split == -1:
            return exception_node
        split = (X[best_split] + X[best_split - 1]) / 2
        left_node = self.fit(X[:best_split], y[:best_split], depth + 1, regression)
        right_node = self.fit(X[best_split:], y[best_split:], depth + 1, regression)
        return DecisionTreeNode(split, left_node, right_node)

    def predict(self, root, val):
        while root.left or root.right:
            if val <= root.val:
                root = root.left
            else:
                root = root.right
        return root.val


def main():
    inputs, targets, _, _ = load_data()
    tree = DecisionTree(5)
    root = tree.fit(inputs, targets)
    pred = tree.predict(root, 100)
    print(pred)


if __name__ == "__main__":
    main()
