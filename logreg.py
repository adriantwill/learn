from math import e

import torch

from graddesc import grad_desc
from util import load_data

# from torch.utils.data import TensorDataset


def log_loss(pred, target):
    summation = (target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).sum()
    n = target.shape[0]
    return -summation / n


def main():
    inputs, targets, w, b = load_data()
    targets = (targets > 1.797).float()
    curr_pred = 1 / (1 + e ** -((inputs * w) + b))
    loss = log_loss(curr_pred, targets)
    print(f"inital loss {loss}")
    for _ in range(10):
        w, b = grad_desc(loss, w, b, 0.01)
        curr_pred = 1 / (1 + e ** -((inputs * w) + b))
        loss = log_loss(curr_pred, targets)
        print(f"updated loss {loss}")
    final_pred = (curr_pred > 0.5).float()
    print("final pred:", final_pred)
    print("true vals:", targets)


if __name__ == "__main__":
    main()
