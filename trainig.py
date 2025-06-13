import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from ldce_model import LDCE_Net

# Configs
NUM_CLASSES = 3  
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_PATH = 'C:\\Users\\FALCON JNB\\Downloads\\LDCE_Net_Model\\Liver Ultrasounds'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = 'ldce_model.pt'

# Transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset
full_dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = LDCE_Net(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Training Loop
train_acc, val_acc, train_loss, val_loss = [], [], [], []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
    train_loss.append(total_loss / len(train_loader))
    train_acc.append(100. * correct / len(train_loader.dataset))

    model.eval()
    val_loss_epoch, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = model(images)
            loss = criterion(output, labels)
            val_loss_epoch += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss.append(val_loss_epoch / len(val_loader))
    val_acc.append(100. * correct / len(val_loader.dataset))

    print(f"Epoch {epoch+1}: Train Loss={train_loss[-1]:.4f}, Val Loss={val_loss[-1]:.4f}, Train Acc={train_acc[-1]:.2f}%, Val Acc={val_acc[-1]:.2f}%")

    if val_loss[-1] < best_val_loss:
        best_val_loss = val_loss[-1]
        torch.save(model.state_dict(), SAVE_PATH)

# Evaluation Metrics
print("\nEvaluating on validation set...")
conf_mat = confusion_matrix(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
acc = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {acc*100:.2f}%")
print(f"F1 Score: {f1:.4f}")

# Save Metrics Plots
os.makedirs("plots", exist_ok=True)
plt.figure()
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Loss Curve")
plt.legend()
plt.savefig("plots/loss_curve.png")

plt.figure()
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.savefig("plots/accuracy_curve.png")

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("plots/confusion_matrix.png")

# Save the final model
torch.save(model.state_dict(), SAVE_PATH)   