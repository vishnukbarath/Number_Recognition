import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# =====================================================================
# 1. UPDATE YOUR DATASET PATHS HERE
# =====================================================================
TRAIN_CSV = r"C:\Users\Number_Recognition\data\mnist_train.csv"
TEST_CSV  = r"C:\Users\Number_Recognition\data\mnist_test.csv"

# =====================================================================
# 2. OUTPUT FOLDER SETUP
# =====================================================================
OUTPUT_DIR = "result/mlp/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 3. DATASET CLASS
# =====================================================================
class MNISTDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.labels = df.iloc[:, 0].values
        self.images = df.iloc[:, 1:].values / 255.0  # Flattened already for MLP

        self.images = torch.tensor(self.images, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

# =====================================================================
# 4. LOAD DATA
# =====================================================================
train_dataset = MNISTDataset(TRAIN_CSV)
test_dataset  = MNISTDataset(TEST_CSV)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# =====================================================================
# 5. MLP MODEL DEFINITION
# =====================================================================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.fc_layers(x)

# =====================================================================
# 6. TRAINING SETUP
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15
loss_history = []
accuracy_history = []

# =====================================================================
# 7. TRAINING LOOP WITH PROGRESS BAR
# =====================================================================
print("\nTraining started...\n")
training_start = torch.cuda.Event(enable_timing=True)
training_start.record()

for epoch in range(num_epochs):
    epoch_loss = 0
    correct = 0
    total = 0

    model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100 * correct / total

    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}  Accuracy={accuracy:.2f}%")

# =====================================================================
# 8. SAVE TRAINING CURVES
# =====================================================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title("MLP Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(accuracy_history)
plt.title("MLP Training Accuracy (%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"))
plt.show()

# =====================================================================
# 9. TEST ACCURACY
# =====================================================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

# =====================================================================
# 10. SAVE A SAMPLE PREDICTION IMAGE
# =====================================================================
sample_img, sample_label = test_dataset[0]
sample_img_gpu = sample_img.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(sample_img_gpu)
    _, pred = torch.max(output, 1)

plt.imshow(sample_img.view(28,28).cpu(), cmap='gray')
plt.title(f"Predicted: {pred.item()} | Actual: {sample_label}")
plt.savefig(os.path.join(OUTPUT_DIR, "sample_prediction.png"))
plt.show()

print(f"Sample prediction saved at: {OUTPUT_DIR}/sample_prediction.png")

# =====================================================================
# 11. SAVE MODEL
# =====================================================================
MODEL_PATH = "result/mlp/mlp_model.pth"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"Trained MLP model saved at: {MODEL_PATH}")
