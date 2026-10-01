import torch
from sklearn.datasets import fetch_california_housing


# from torch.utils.data import TensorDataset
def mse(y, y_hat):
    squared_errors = (y - y_hat) ** 2
    summation = squared_errors.sum()
    n = y.shape[0]
    return summation / n


def grad_desc(loss: torch.Tensor, w: torch.Tensor, b: torch.Tensor, lr: float):
    loss.backward()
    new_w = w - lr * w.grad
    new_b = b - lr * b.grad
    return inputs * new_w + new_b


X_housing, y_housing = fetch_california_housing(return_X_y=True)
X_housing = X_housing[:, 0]
inputs = torch.from_numpy(X_housing)
targets = torch.from_numpy(y_housing)
# dataset=TensorDataset(inputs,targets)
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
curr_pred = inputs * w + b
loss = mse(curr_pred, targets)
print(f"inital loss {loss}")
updated_pred = grad_desc(loss, w, b, 0.1)
updated_loss = mse(updated_pred, targets)
print(f"updated loss {updated_loss}")
