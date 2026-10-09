import torch


def grad_desc(loss: torch.Tensor, w: torch.Tensor, b: torch.Tensor, lr: float):
    w.grad = None
    b.grad = None
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    return w, b


def main():
    pass


if __name__ == "__main__":
    main()
