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

X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



class LinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1),requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1),requires_grad=True)

    def forward(self, x):
        return self.weights*x + self.bias 



Model_0 = LinearRegressionModel()

Loss = nn.L1Loss()
optimizer = torch.optim.SGD(params=Model_0.parameters(), lr=0.01)

epochs = 300

for epoch in range(epochs):
    
    Model_0.train()

    y_preds = Model_0(X_train)

    Loss_value = Loss(y_preds, y_train)

    optimizer.zero_grad()

    Loss_value.backward()

    optimizer.step()

    Model_0.eval()
    with torch.inference_mode():
        test_preds = Model_0(X_test)
        test_loss = Loss(test_preds, y_test)

    
    # if epoch % 30 == 0:
    #     print(f"Epoch: {epoch} || Loss: {Loss_value.item()} || Test Loss: {test_loss.item()}")


plt.figure(figsize=(8, 6))
plt.scatter(X_train.cpu().numpy(), y_train.cpu().numpy(),c="b", label='Training data')
plt.scatter(X_test.cpu().numpy(), y_test.cpu().numpy(),c="r" ,label='Testing data')
plt.scatter(X_train.cpu().numpy(), y_preds.detach().cpu(),c="g", label='Testing data')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Scatter plot of training and testing data')
plt.show()

print(Model_0.state_dict())