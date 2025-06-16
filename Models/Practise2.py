import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

weight = 0.3
bias = 0.9

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1).to(device)
Y = weight * X + bias
Y = Y.to(device)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

plt.figure(figsize=(8, 6))
plt.scatter(X_train.cpu().numpy(), y_train.cpu().numpy(), label='Training data')
plt.scatter(X_test.cpu().numpy(), y_test.cpu().numpy(), label='Testing data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Scatter plot of training and testing data')
plt.show()

class LinearRegressionModel(nn.Module):

    def __init__(self,intput_features=1, output_features=1):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1),requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1),requires_grad=True)

    def forward(self, x):
        return self.weights*X + self.bias 


Model_0 = LinearRegressionModel(X_train).to(device)
print(Model_0.state_dict())