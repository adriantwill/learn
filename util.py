import pandas as pd
import torch


def load_data():
    X_housing = pd.read_csv("housing.csv", usecols=["MedInc"]).to_numpy()
    y_housing = pd.read_csv("housing.csv", usecols=["MedHouseVal"]).to_numpy()
    inputs = torch.from_numpy(X_housing)
    targets = torch.from_numpy(y_housing)
    # dataset=TensorDataset(inputs,targets)
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    return inputs, targets, w, b


def main():
    X, y, w, b = load_data()
    # print(X[0])
    x = torch.randn(4)
    y = torch.randn(4)
    print(torch.dist(x, y, 3.5))


if __name__ == "__main__":
    main()
