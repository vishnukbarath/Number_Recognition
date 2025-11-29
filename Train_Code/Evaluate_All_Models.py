import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision.transforms as transforms
import torchvision.models as models

# =====================================================================
# 1. DATASET PATHS
# =====================================================================
TEST_CSV = r"C:\Users\vishn\Documents\Number_Recognition\data\mnist_test.csv"
BATCH_SIZE = 64

# =====================================================================
# 2. OUTPUT FOLDER
# =====================================================================
OUTPUT_DIR = "result/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 3. DATASET CLASSES
# =====================================================================
# CNN / Advanced CNN Dataset (28x28, 1 channel)
class MNISTDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.labels = df.iloc[:,0].values
        images = df.iloc[:,1:].values.reshape(-1,1,28,28) / 255.0
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ResNet Dataset (224x224)
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
# 4. LOAD TEST DATA
# =====================================================================
test_dataset_cnn = MNISTDataset(TEST_CSV)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=BATCH_SIZE, shuffle=False)

test_dataset_mlp = MNISTDataset(TEST_CSV)  # MLP uses flattened
test_loader_mlp = DataLoader(test_dataset_mlp, batch_size=BATCH_SIZE, shuffle=False)

test_dataset_resnet = MNISTResNetDataset(TEST_CSV)
test_loader_resnet = DataLoader(test_dataset_resnet, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================================
# 5. DEFINE MODEL CLASSES
# =====================================================================
# CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128*7*7,512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512,10)
        )
    def forward(self,x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# Advanced CNN
class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*7*7,512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,10)
        )
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

# MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(28*28,512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x = x.view(x.size(0),-1)
        return self.fc_layers(x)

# ResNet18
def get_resnet():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
    model.fc = nn.Linear(model.fc.in_features,10)
    return model

# =====================================================================
# 6. HELPER FUNCTION FOR TESTING
# =====================================================================
def evaluate_model(model, test_loader, device, model_name):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = 100*correct/total
    print(f"{model_name} Test Accuracy: {acc:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR,f"{model_name}_confusion_matrix.png"))
    plt.close()

    # Save a few example predictions
    for i in range(5):
        img = test_loader.dataset[i][0]
        label = test_loader.dataset[i][1]
        pred = all_preds[i]
        plt.imshow(img.squeeze().cpu(), cmap='gray')
        plt.title(f"True: {label} Pred: {pred}")
        plt.savefig(os.path.join(OUTPUT_DIR,f"{model_name}_sample_{i}.png"))
        plt.close()

    return acc

# =====================================================================
# 7. MAIN TEST LOOP
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load all models
cnn_model = CNN()
cnn_model.load_state_dict(torch.load("result/cnn_model/cnn_model.pth", map_location=device))

adv_cnn_model = AdvancedCNN()
adv_cnn_model.load_state_dict(torch.load("result/advanced_cnn/advanced_cnn_model.pth", map_location=device))

mlp_model = MLP()
mlp_model.load_state_dict(torch.load("result/mlp/mlp_model.pth", map_location=device))

resnet_model = get_resnet()
resnet_model.load_state_dict(torch.load("result/resnet/resnet_mnist.pth", map_location=device))

# Evaluate
print("\nEvaluating CNN...")
cnn_acc = evaluate_model(cnn_model, test_loader_cnn, device, "CNN")

print("\nEvaluating Advanced CNN...")
adv_cnn_acc = evaluate_model(adv_cnn_model, test_loader_cnn, device, "Advanced_CNN")

print("\nEvaluating MLP...")
mlp_acc = evaluate_model(mlp_model, test_loader_mlp, device, "MLP")

print("\nEvaluating ResNet18...")
resnet_acc = evaluate_model(resnet_model, test_loader_resnet, device, "ResNet18")

# =====================================================================
# 8. SUMMARY
# =====================================================================
summary_text = f"""
Model Evaluation Summary:

CNN Accuracy: {cnn_acc:.2f}%
Advanced CNN Accuracy: {adv_cnn_acc:.2f}%
MLP Accuracy: {mlp_acc:.2f}%
ResNet18 Accuracy: {resnet_acc:.2f}%
"""

print(summary_text)
with open(os.path.join(OUTPUT_DIR,"evaluation_summary.txt"), "w") as f:
    f.write(summary_text)
