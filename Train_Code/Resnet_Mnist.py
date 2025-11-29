import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchvision.models as models

# =====================================================================
# 1. DATASET PATHS
# =====================================================================
TRAIN_CSV = r"C:\Users\vishn\Documents\Number_Recognition\data\mnist_train.csv"
TEST_CSV  = r"C:\Users\vishn\Documents\Number_Recognition\data\mnist_test.csv"

# =====================================================================
# 2. OUTPUT FOLDER
# =====================================================================
OUTPUT_DIR = "result/resnet/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 3. DATASET CLASS WITH RESIZING
# =====================================================================
class MNISTResNetDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.labels = df.iloc[:,0].values
        images = df.iloc[:,1:].values.reshape(-1,28,28).astype('uint8') 


        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        self.images = torch.stack([transform(img) for img in images])
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# =====================================================================
# 4. LOAD DATA
# =====================================================================
train_dataset = MNISTResNetDataset(TRAIN_CSV)
test_dataset  = MNISTResNetDataset(TEST_CSV)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# =====================================================================
# 5. MODEL SETUP (RESNET18)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights="IMAGENET1K_V1")
model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)  # single channel
model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5  # can increase if needed
loss_history = []
accuracy_history = []

# =====================================================================
# 6. TRAINING LOOP
# =====================================================================
print("\nTraining started...\n")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100*correct/total
    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} Accuracy={accuracy:.2f}%")

# =====================================================================
# 7. SAVE TRAINING CURVES
# =====================================================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title("ResNet18 Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(accuracy_history)
plt.title("ResNet18 Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.savefig(os.path.join(OUTPUT_DIR,"training_curve.png"))
plt.show()

# =====================================================================
# 8. TEST ACCURACY
# =====================================================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

test_acc = 100*correct/total
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

# =====================================================================
# 9. SAMPLE PREDICTION IMAGE
# =====================================================================
sample_img, sample_label = test_dataset[0]
sample_img_gpu = sample_img.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(sample_img_gpu)
    _, pred = torch.max(output,1)

plt.imshow(sample_img.squeeze(), cmap='gray')
plt.title(f"Predicted: {pred.item()} | Actual: {sample_label}")
plt.savefig(os.path.join(OUTPUT_DIR,"sample_prediction.png"))
plt.show()

# =====================================================================
# 10. SAVE MODEL
# =====================================================================
MODEL_PATH = "result/resnet/resnet_mnist.pth"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"Trained ResNet18 model saved at: {MODEL_PATH}")
