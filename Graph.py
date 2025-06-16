import torch
import matplotlib.pyplot as plt
from torch import nn

weight = 0.7
bias = 0.2

start = 0
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias
print(X[:10])
print("\n")
print(y[:10])

split = int(0.8 * len(X))
x_train,y_train = X[:split],y[:split]
x_test,y_test = X[split:],y[split:]
len(x_train),len(y_train),len(x_test),len(y_test)

def plot_predictions(train_data = x_train,train_label = y_train,test_data = x_test,test_label = y_test,predictions= None):
    plt.figure(figsize = (10,7))

    plt.scatter(train_data,train_label,c="b",s=4,label = "Training Data")
    plt.scatter(test_data,test_label,c="g",s=4,label = "Test Data")
    if predictions is not None:
        plt.scatter(test_data,predictions,c="r",s=4,label = "Predictions")
    plt.legend(prop={"size":14})
    plt.show()

plot_predictions()