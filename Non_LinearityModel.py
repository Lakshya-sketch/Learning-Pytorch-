import torch.nn as nn
import torch
import numpy 
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"


#---------------------------CREATING-DATA--------------------------------------------------------------------

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42



X_blob, Y_blob = make_blobs(n_samples=1000,
                            n_features = NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std = 1.5,
                            random_state = RANDOM_SEED)

X_blob = torch.from_numpy(X_blob).type(torch.float)
Y_blob = torch.from_numpy(Y_blob).type(torch.float)



X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        Y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

plt.figure(figsize=(10, 7))
plt.scatter(X_blob_train[:, 0], X_blob_train[:, 1], c=y_blob_train, cmap=plt.cm.RdYlBu)


#-----------------------------------------------------------------------------------------------


#---------------------------CREATING-MULTICLASS-MODEL--------------------------------------------------------------------       

class BlobModel(nn.Module):
    def __init__(self,input_features,output_features,hidden_units = 8):
        """Initial multi-class classification model.
        
        Args:
            input_features (int): Number of input features.
            output_features (int): Number of output features.
            hidden_units (int, optional): Number of hidden units. Defaults to 8.

        Returns:

        Example:  
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self,x):
        return self.linear_layer_stack(x)
    
model_4 = BlobModel(input_features=2,
                    output_features=4,
                    hidden_units=8).to(device)

#Creating Loss Function and optimizer

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr = 0.1)

model_4(X_blob_test.to(device))

model_4.eval()
with torch.inference_mode():
    y_logits= model_4(X_blob_test.to(device))


y_pred_probs = torch.softmax(y_logits, dim=1)
print("Y_logits are ",y_logits[:5],"\n")
print("Y_pred_probs are ",y_pred_probs[:5],"\n")


torch.manual_seed(42)
torch.cuda.manual_seed(42)
#---------------------------TRAINING-LOOP--------------------------------------------------------------------

epochs = 100

X_blob_train = X_blob_train.to(device)
y_blob_train = y_blob_train.to(device)

X_blob_test = X_blob_test.to(device)
y_blob_test = y_blob_test.to(device)

for epoch in range(epochs):
    model_4.train()

    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim = 1).argmax(dim = 1)

    loss = loss_fn(y_logits,y_blob_train)
    acc = accuracy_fn(y_true = y_blob_train,
                      y_prep=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    