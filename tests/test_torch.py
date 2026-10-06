import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

y = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0]
])

result = x + y

print("Tensor X:")
print(x)

print("Tensor Y:")
print(y)

print("X + Y:")
print(result)
