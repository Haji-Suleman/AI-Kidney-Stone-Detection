import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
df = pd.read_csv("./kidney-stone-dataset.csv")
df = df.drop(columns=["Unnamed: 0"])

X = df[["ph", "osmo", "urea", "calc", "gravity", "cond"]].values
y = df["target"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to torch tensors
X_scaled = torch.tensor(X_scaled, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
)


# Model
class KidneyStoneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(6, 8)
        self.layer2 = nn.Linear(8, 4)
        self.layer3 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x  # raw logits


model = KidneyStoneModel().to(device)

# Loss & optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 200
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(epochs):
    # Train
    model.train()
    logits = model(X_train).squeeze()
    loss = loss_fn(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Eval
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_loss = loss_fn(test_logits, y_test)

        # Convert logits to probabilities
        train_preds = torch.round(torch.sigmoid(logits))
        test_preds = torch.round(torch.sigmoid(test_logits))

        train_acc = (train_preds == y_train).float().mean().item()
        test_acc = (test_preds == y_test).float().mean().item()

    if epoch % 10 == 0:
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(
            f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss.item():.4f}, Test Acc: {test_acc:.4f}"
        )

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(range(0, epochs, 10), train_losses, label="Train Loss")
plt.plot(range(0, epochs, 10), test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(0, epochs, 10), train_accs, label="Train Acc")
plt.plot(range(0, epochs, 10), test_accs, label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
