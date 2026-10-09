
from graddesc import grad_desc
from util import load_data


# from torch.utils.data import TensorDataset
def mse(y, y_hat):
    summation = ((y - y_hat) ** 2).sum()
    n = y.shape[0]
    return summation / n


def main():
    inputs, targets, w, b = load_data()
    curr_pred = inputs * w + b
    loss = mse(curr_pred, targets)
    print(f"inital loss {loss}")
    for _ in range(10):
        w, b = grad_desc(loss, w, b, 0.01)
        updated_pred = inputs * w + b
        updated_loss = mse(updated_pred, targets)
        loss = updated_loss
        print(f"updated loss {updated_loss}")
    final_pred = inputs * w + b
    print("final pred:", final_pred)
    print("true vals:", targets)


if __name__ == "__main__":
    main()
