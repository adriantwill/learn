import torch

from graddesc import grad_desc
from util import load_data


# from torch.utils.data import TensorDataset
def normal_equation(X, y):
    X = torch.column_stack((torch.ones(len(X)), X))
    return torch.linalg.inv(X.T @ X) @ X.T @ y


def mse(y_hat, y):
    summation = ((y - y_hat) ** 2).sum()
    n = y.shape[0]
    return summation / n


def ridge(w, alpha, y):
    n = y.shape[0]
    return (alpha / n) * (w**2).sum()


def lasso(w, alpha):
    return 2 * alpha * (w.abs()).sum()


def elastic_net(r, w, alpha, y):
    ridge_reg = ridge(w, alpha, y)
    lasso_reg = lasso(w, alpha)
    return r * lasso_reg + (1 - r) * ridge_reg


def main():
    inputs, targets, w, b = load_data()
    for _ in range(10):
        predictions = inputs * w + b
        loss = mse(predictions, targets) + ridge(w, 0.5, targets)
        w, b = grad_desc(loss, w, b, 0.01)
        print(f"updated loss: {loss}")
    final_pred = inputs * w + b
    print("final pred:", final_pred)
    print("true vals:", targets)


if __name__ == "__main__":
    main()
