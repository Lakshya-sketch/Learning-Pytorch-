import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torchmetrics

# 1. DATA GENERATION (No changes needed here)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

# 2. DATA CONVERSION (CORRECTION #1)
# Input features X must be float, target labels y must be long
X = torch.from_numpy(X).to(torch.float32)
y = torch.from_numpy(y).to(torch.long)

# 3. DATA SPLITTING & DEVICE SETUP
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move all data to the target device once for efficiency (CORRECTION #4)
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

# 4. MODEL DEFINITION (No changes needed)
class SpiralClassifier(nn.Module):
    def __init__(self, input_features=2, hidden_units=8, output_features=3):
       super().__init__()
       self.linear_layer_stack = nn.Sequential(
          nn.Linear(in_features=input_features, out_features=hidden_units),
          nn.ReLU(),
          nn.Linear(in_features=hidden_units, out_features=hidden_units),
          nn.ReLU(),
          nn.Linear(in_features=hidden_units, out_features=output_features),
       )
    
    def forward(self, x):
       return self.linear_layer_stack(x)

# 5. MODEL INITIALIZATION
model = SpiralClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=K).to(device)

epochs = 200
for epoch in range(epochs):
    ### Training
    model.train()

    # Forward pass
    y_logits = model(x_train)

    # Calculate loss
    loss = loss_fn(y_logits, y_train)
   
    # Optimizer zero grad, backward pass, and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### Evaluation (every 10 epochs)
    if epoch % 10 == 0:
        model.eval()
        with torch.inference_mode():
            # Forward pass on test data
            test_logits = model(x_test)
            
            # Calculate test loss
            test_loss = loss_fn(test_logits, y_test)
            
            # Get predictions using the correct method for multi-class (CORRECTION #2)
            test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

            # Calculate accuracy
            test_acc = accuracy_fn(test_preds, y_test)

            print(f"Epoch: {epoch} | Train Loss: {loss:.5f} | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc*100:.2f}%")

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.title("Spiral Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
