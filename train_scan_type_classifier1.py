import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT  # or .IMAGENET1K_V1 if needed
model = resnet18(weights=weights)


# Step 1: Path setup
data_dir = r"C:\\Users\\Prath\\OneDrive\\Desktop\\FinalProject\\code\\FinalApp(all images)\\image_dataset"# same structure: ultrasound/, mri/, lungs/, breast/
model_path = "scan_type_classifier.pth"

# Step 2: Transform setup

transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])




# Step 3: Load dataset and split into train/val
from sklearn.model_selection import train_test_split
dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = dataset.classes
indices = list(range(len(dataset)))
targets = [dataset.imgs[i][1] for i in indices]
train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=targets, random_state=42)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Step 4: Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.to(device)


# Step 5: Train with validation
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
num_epochs = 10
best_val_acc = 0.0
print(" Training started...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # Validation
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_acc = correct / total
    print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': class_names
        }, 'scan_type_classifier.pth')

# Step 6: Final evaluation on validation set
from sklearn.metrics import confusion_matrix, classification_report
print("\nValidation Results (Best Model):")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=class_names))

print(f"[OK] Model saved as scan_type_classifier.pth with classes: {class_names}")