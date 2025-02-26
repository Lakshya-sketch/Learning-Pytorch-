import torch
import matplotlib.pyplot as plt
from torch import nn

# Creating dataset
weight = 0.7
bias = 0.2

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)  # Creating input data
y = weight * X + bias  # Generating corresponding labels

# Splitting into training and testing sets (80% train, 20% test)
split = int(0.8 * len(X))
x_train, y_train = X[:split], y[:split]
x_test, y_test = X[split:], y[split:]

# Defining Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
# Function to plot predictions
def plot_predictions(train_data=x_train, train_label=y_train, 
                     test_data=x_test, test_label=y_test, 
                     predictions=None):
    plt.figure(figsize=(10, 7))

    # Scatter plot for training data (blue)
    plt.scatter(train_data, train_label, c="b", s=4, label="Training Data")

    # Scatter plot for testing data (green)
    plt.scatter(test_data, test_label, c="g", s=4, label="Test Data")

    # Added: Scatter plot for predictions (red)
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size":14})
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Model Predictions")
    plt.show()

# Setting random seed for reproducibility
torch.manual_seed(42)
model_0 = LinearRegressionModel()

# Making predictions before training
with torch.inference_mode():
    y_preds = model_0(x_test)

# Setting up Loss and Optimizer
loss = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001)

# Training Loop
epochs = 100
for epoch in range(epochs):
    model_0.train()  # Set model to training mode

    # 1. Forward pass
    y_preds = model_0(x_train)

    # 2. Compute loss
    loss_val = loss(y_preds, y_train)

    # 3. Zero gradients
    optimizer.zero_grad()

    # 4. Backward pass
    loss_val.backward()

    # 5. Update weights
    optimizer.step()

    # Print loss for monitoring
    print(f"Epoch {epoch+1}, Loss: {loss_val.item()}")

# Making new predictions after training
with torch.inference_mode():
    y_preds_new = model_0(x_test)

print("Final Model Parameters:", model_0.state_dict())
plot_predictions(predictions=y_preds_new)
plot_predictions(predictions=y_preds)
