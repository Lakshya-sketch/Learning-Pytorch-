# Importing Necessary Libraries
import torch
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path

# Creating Data
weight = 0.6
bias = 0.4

# Creating range
start = 0
end = 3
step = 0.02

# Creating x and y values
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = x * weight + bias

# Split data
train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

# Define Plot Function
def plot_predictions(train_data=x_train,
                     train_labels=y_train,
                     test_data=x_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(12, 9))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})  
    plt.show()  

# Print dataset sizes
print(len(x_train), len(y_train), len(x_test), len(y_test))

# Making a Linear Regression Model
class LinearRegressionModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
# Setting Manual Seed
torch.manual_seed(42)
model_0 = LinearRegressionModelV1()
print(model_0.state_dict())

# Making a Training Loop
loss_fn = nn.L1Loss()  # Renamed to avoid overwriting
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

torch.manual_seed(42)

epochs = 201
for epoch in range(epochs):  # Fixed the loop
    model_0.train()

    # 1- Forward Pass
    y_preds = model_0(x_train)

    # 2- Calculate Loss
    loss_val = loss_fn(y_preds, y_train)  # Fixed loss calculation

    # 3- Zero Gradients
    optimizer.zero_grad()

    # 4- Backward Pass
    loss_val.backward()

    # 5- Optimizer Step
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(x_test)
        test_loss = loss_fn(test_pred, y_test)  # Fixed loss function usage

    # Logging every 10 epochs
    # if epoch % 10 == 0:  # Fixed condition
    #     print(f"Epoch {epoch} || Loss: {loss_val.item()} || Test Loss: {test_loss.item()}")

model_0.eval()

with torch.inference_mode():
    y_preds = model_0(x_test)

# plot_predictions(predictions=y_preds)

#Saving a Model
MODEL_PATH = Path("Models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

MODEL_NAME = "Model_0"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(obj=model_0.state_dict(),
            f=MODEL_SAVE_PATH.with_suffix(".pth"))  

print(MODEL_SAVE_PATH)