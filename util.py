import pandas as pd
import torch


def load_data():
    X_housing = pd.read_csv("housing.csv", usecols=["MedInc"]).to_numpy()
    y_housing = pd.read_csv("housing.csv", usecols=["MedHouseVal"]).to_numpy()
    inputs = torch.from_numpy(X_housing)
    print(inputs)
    targets = torch.from_numpy(y_housing)
    # dataset=TensorDataset(inputs,targets)
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    return inputs, targets, w, b


def main():
    load_data()


if __name__ == "__main__":
    main()
