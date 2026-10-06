import torch
import torch.nn as nn

# CPU par run karenge
device = torch.device("cpu")

# Simple model:
# y = wx + b
model = nn.Linear(1, 1).to(device)

# Training data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], device=device)

# Loss function
loss_fn = nn.MSELoss()

# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

print("Before training:")

with torch.no_grad():
    print(model(x))

# Training loop
for epoch in range(1000):

    # Forward pass
    prediction = model(x)

    # Calculate loss
    loss = loss_fn(prediction, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.6f}"
        )

print("\nAfter training:")

with torch.no_grad():
    print(model(x))
