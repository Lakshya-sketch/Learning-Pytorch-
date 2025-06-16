import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torchmetrics
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

X,Y = make_moons(n_samples = 500,random_state = 42)

X = torch.from_numpy(X).type(torch.float)
Y = torch.from_numpy(Y).type(torch.float)

Y = Y.unsqueeze(dim = 1) # or Y.flatten() # Reshape Y to be a column vector

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

X_train, X_test = X_train.to(device), X_test.to(device)
Y_train, Y_test = Y_train.to(device), Y_test.to(device)


class MoonsModel(nn.Module):
    def __init__(self, input_features=2, output_features=1, hidden_units=8):
        """Initial multi-class classification model.
        
        Args:
            input_features (int): Number of input features.
            output_features (int): Number of output features.
            hidden_units (int, optional): Number of hidden units. Defaults to 8.
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
    
            nn.Linear(in_features=hidden_units, out_features=hidden_units), # Corrected: out_features=hidden_units
            nn.ReLU(),
            
            nn.Linear(in_features=hidden_units, out_features=hidden_units), # Corrected: out_features=hidden_units
            nn.ReLU(),
            
            nn.Linear(in_features=hidden_units, out_features=output_features)

        )

    def forward(self, x):
        return self.linear_layer_stack(x)



Model_1 = MoonsModel().to(device) 

Loss = nn.BCEWithLogitsLoss()
Optimizer = torch.optim.Adam(params=Model_1.parameters(), lr=0.01)

epochs = 200

for epcoh in range(epochs):
    
    Model_1.train()

    y_logits_train = Model_1(X_train.to(device))

    Loss_value = Loss(y_logits_train, Y_train.to(device))

    Optimizer.zero_grad()

    Loss_value.backward()

    Optimizer.step()

    if epcoh % 10 == 0:
        Model_1.eval()
        with torch.inference_mode():
            y_logits_test = Model_1(X_test.to(device))
            test_loss = Loss(y_logits_test, Y_test.to(device))
            
            y_preds_test = torch.round(torch.sigmoid(y_logits_test))

            acc = torchmetrics.Accuracy(task="binary", threshold=0.5).to(device)
            print(f"Epoch: {epcoh} | Train Loss: {Loss_value:.5f} | Test Loss: {test_loss:.5f} || Test Accuracy: {acc(y_preds_test, Y_test.to(device))*100:.2f}%")

Model_1.eval()
with torch.inference_mode():
    test_logits = Model_1(X_test)
    test_probs = torch.sigmoid(test_logits)
    y_test_pred_labels = (test_probs > 0.5).float()


X_train_cpu = X_train.cpu().numpy()
Y_train_cpu = Y_train.cpu().numpy().squeeze()
X_test_cpu = X_test.cpu().numpy()
Y_test_cpu = Y_test.cpu().numpy().squeeze()
y_test_pred_labels_cpu = y_test_pred_labels.cpu().numpy().squeeze()

# Function to plot the decision boundary
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - CS231n Stanford Lecture notes on PyTorch (adapted)
    """
    # Put everything to CPU (works best with NumPy and Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    # Create a meshgrid (a grid of points)
    # h is the step size in the grid
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Make predictions on the grid
    # 1. Stack the xx and yy coordinates into a 2D array where each row is a point (x, y)
    #    np.c_ is a helper to concatenate arrays column-wise
    #    xx.ravel() flattens the xx grid into a 1D array
    grid_points_np = np.c_[xx.ravel(), yy.ravel()]
    # 2. Convert these NumPy points to a PyTorch tensor
    grid_points_torch = torch.from_numpy(grid_points_np).float()

    # 3. Put the model in evaluation mode
    model.eval()
    # 4. Get predictions (logits)
    with torch.inference_mode():
        y_logits = model(grid_points_torch)

    # 5. Convert logits to predicted labels (0 or 1)
    #    Apply sigmoid to get probabilities, then threshold at 0.5
    y_pred_labels = torch.round(torch.sigmoid(y_logits))

    # Reshape predictions to match the xx grid shape for plotting
    # .numpy() is needed because plt.contourf expects NumPy arrays
    # .squeeze() removes the trailing dimension if y_pred_labels is (N,1)
    zz = y_pred_labels.numpy().reshape(xx.shape).squeeze()

    # Plot the decision boundary using contourf
    plt.figure(figsize=(10, 7))
    # contourf fills the regions with colors
    plt.contourf(xx, yy, zz, cmap=plt.cm.RdYlBu, alpha=0.7)

    # Scatter plot the original data points (X)
    # Color them by their true labels (y)
    # .squeeze() is used on y to make sure it's 1D for coloring
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), s=40, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()

# --- How to use the function ---
# Plot decision boundary for the training data
print("\nPlotting decision boundary for training data...")
plot_decision_boundary(Model_1, X_train, Y_train)
