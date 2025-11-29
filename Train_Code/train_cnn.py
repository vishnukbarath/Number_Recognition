import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import time


# =====================================================================
# 1. UPDATE YOUR DATASET PATHS HERE
# =====================================================================
TRAIN_CSV = r"C:\Users\vishn\Documents\Number_Recognition\data\mnist_train.csv"
TEST_CSV  = r"C:\Users\vishn\Documents\Number_Recognition\data\mnist_test.csv"


# =====================================================================
# 2. DATASET CLASS (LOADS MNIST CSV)
# =====================================================================
class MNISTDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.labels = df.iloc[:, 0].values
        self.images = df.iloc[:, 1:].values.reshape(-1, 1, 28, 28) / 255.0

        self.images = torch.tensor(self.images, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


# =====================================================================
# 3. LOAD DATA
# =====================================================================
train_dataset = MNISTDataset(TRAIN_CSV)
test_dataset  = MNISTDataset(TEST_CSV)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


# =====================================================================
# 4. CNN MODEL (POWERFUL + HIGH ACCURACY)
# =====================================================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        out = self.fc_layers(x)
        return out


# =====================================================================
# 5. TRAINING SETUP
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

loss_history = []
acc_history = []


# =====================================================================
# 6. TRAINING LOOP
# =====================================================================
print("\nTraining started...\n")
total_start = time.time()

for epoch in range(num_epochs):
    start = time.time()
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    loss_history.append(epoch_loss)
    acc_history.append(epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Loss: {epoch_loss:.4f} | "
          f"Accuracy: {epoch_acc:.2f}% | "
          f"Time: {time.time() - start:.2f}s")

print(f"\nTotal Training Time: {time.time() - total_start:.2f}s")


# =====================================================================
# 7. PLOT TRAINING CURVES
# =====================================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, label="Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(acc_history, label="Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.show()


# =====================================================================
# 8. TESTING ACCURACY
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
# 9. SAMPLE PREDICTION
# =====================================================================
sample_img, sample_label = test_dataset[0]
sample_img_gpu = sample_img.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(sample_img_gpu)
    _, pred = torch.max(output, 1)

print(f"\nSample Image Prediction: {pred.item()} (Actual: {sample_label})")

plt.imshow(sample_img.squeeze(), cmap='gray')
plt.title(f"Predicted: {pred.item()}")
plt.show()
