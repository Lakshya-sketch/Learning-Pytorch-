import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

#Make 1000 Samples
n_samples = 1000

#Making The Data
X, Y = make_circles(n_samples,
                   noise=0.05,
                   random_state=42)

#Converting Data From Numpy Array to Tensor
x = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(Y).type(torch.float32)

x_train, x_test, y_train, y_test = train_test_split(x,
                                                 y,
                                                 test_size=0.2,
                                                 random_state=42)

#Sending Data to GPU
x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

#Construct A Model That Subclasses nn.Module
class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        #Creating 2 layers
        self.layer_1 = nn.Linear(in_features=2, out_features=16) #Taking in 2 inputs and Giving out 8.
        self.layer_2 = nn.Linear(in_features=16, out_features=16) #Taking same shape as input which was outputted by layer 1
        self.layer_3 = nn.Linear(in_features=16, out_features=16)
        self.layer_4 = nn.Linear(in_features=16, out_features=1) #Taking

    def forward(self, x):
        return self.layer_4(self.layer_3(self.layer_2(self.layer_1(x))))
    
circle_model = CircleModel().to(device)

#Setting Up Loss function
Loss = nn.BCEWithLogitsLoss()

#Using Optimizer
optimizer = torch.optim.SGD(params=circle_model.parameters(), lr=0.1)

def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

#Building Training Loop
epochs = 1001

torch.manual_seed(42)
torch.cuda.manual_seed(42)

for epoch in range(epochs):
    #Training
    circle_model.train()
    
    #Forward Pass
    y_logits = circle_model(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) #Turn Logits --> preds prob --> pred label
    
    #Calculate Loss/Acc 
    loss = Loss(y_logits, y_train)
    acc = accuracy(y_train, y_pred)
    
    #Optimizer
    optimizer.zero_grad()

    #Backpropagation
    loss.backward()

    #Gradient Descent
    optimizer.step()

    circle_model.eval()
    with torch.inference_mode():
        test_logits = circle_model(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

    #Calculate Test loss/Acc
    test_loss = Loss(test_logits, y_test)
    test_acc = accuracy(y_test, test_pred)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} || Loss: {loss:.5f} , Acc {acc:.2f} |\n| Test Loss: {test_loss:.5f} , Test Acc {test_acc:.2f} \n")


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train ")
plot_decision_boundary(circle_model, x_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(circle_model, x_test, y_test)

plt.show()