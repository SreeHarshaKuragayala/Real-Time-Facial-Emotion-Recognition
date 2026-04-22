import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time
from multiprocessing import freeze_support

# =============================================================================
# CONFIGURATION
# =============================================================================
# Paths
train_dir = "F:/Python/Real-Time-Facial-Emotion-Recognition/DATASET/train"
test_dir = "F:/Python/Real-Time-Facial-Emotion-Recognition/DATASET/test"

# Settings optimized for RTX 4060 8GB
img_size = 224
batch_size = 64  # Optimized for RTX 4060
num_classes = 8
learning_rate = 0.001
epochs_phase1 = 50
epochs_phase2 = 100


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class CustomCNN(nn.Module):
    """Custom CNN Architecture"""

    def __init__(self, num_classes=8):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Second block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Third block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Fourth block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_efficientnet_model(num_classes=8):
    """Create EfficientNet-B0 model"""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze base model initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )

    return model


def create_resnet_model(num_classes=8):
    """Create ResNet-50 model"""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze base model initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        pbar.set_postfix({
            'Loss': f'{running_loss / (batch_idx + 1):.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    return running_loss / len(train_loader), 100. * correct / total


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

    return running_loss / len(val_loader), 100. * correct / total


def predict_emotion(image_path, model, device, transform, class_names):
    """Predict emotion from a single image"""
    from PIL import Image

    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    emotion = class_names[predicted.item()]
    confidence = confidence.item()

    return emotion, confidence


def main():
    """Main training function"""
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # Set batch size based on device
    current_batch_size = batch_size

    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        print("⚠️  CUDA not available, using CPU")
        current_batch_size = 32  # Reduce batch size for CPU

    # Create results directory
    os.makedirs("../results", exist_ok=True)

    # =============================================================================
    # DATA PREPROCESSING
    # =============================================================================
    print("🔄 Setting up data preprocessing...")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation and test transforms (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    # Split training data into train and validation (80:20)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    # Create a copy of the dataset for validation with different transform
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(datasets.ImageFolder(train_dir, transform=val_test_transform), val_indices)

    # Create data loaders - REDUCED num_workers for Windows compatibility
    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=current_batch_size, shuffle=False, num_workers=0)

    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    print(f"✅ Test samples: {len(test_dataset)}")
    print(f"✅ Classes: {full_train_dataset.classes}")

    # =============================================================================
    # MODEL SELECTION
    # =============================================================================
    print("🏗️ Creating model...")

    # Choose your model
    # Option 1: Custom CNN
    # model = CustomCNN(num_classes=num_classes)

    # Option 2: EfficientNet-B0 (Recommended)
    model = create_efficientnet_model(num_classes=num_classes)

    # Option 3: ResNet-50
    # model = create_resnet_model(num_classes=num_classes)

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✅ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # =============================================================================
    # TRAINING PHASE 1: FROZEN BASE MODEL
    # =============================================================================
    print("🚀 Starting training phase 1 (frozen base model)...")

    train_losses_1, train_accs_1 = [], []
    val_losses_1, val_accs_1 = [], []
    best_val_acc = 0.0

    start_time = time.time()

    for epoch in range(epochs_phase1):
        print(f"\nEpoch {epoch + 1}/{epochs_phase1}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Update scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"📉 Learning rate reduced to {new_lr:.2e}")

        # Store metrics
        train_losses_1.append(train_loss)
        train_accs_1.append(train_acc)
        val_losses_1.append(val_loss)
        val_accs_1.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '../results/best_model_phase1.pth')
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    phase1_time = time.time() - start_time
    print(f"⏱️ Phase 1 completed in {phase1_time / 60:.1f} minutes")

    # =============================================================================
    # TRAINING PHASE 2: FINE-TUNING
    # =============================================================================
    print("🔥 Starting training phase 2 (fine-tuning)...")

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # Use lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=learning_rate / 10)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_losses_2, train_accs_2 = [], []
    val_losses_2, val_accs_2 = [], []

    start_time = time.time()

    for epoch in range(epochs_phase2):
        print(f"\nEpoch {epoch + 1}/{epochs_phase2}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Update scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"📉 Learning rate reduced to {new_lr:.2e}")

        # Store metrics
        train_losses_2.append(train_loss)
        train_accs_2.append(train_acc)
        val_losses_2.append(val_loss)
        val_accs_2.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '../results/best_model_final.pth')
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    phase2_time = time.time() - start_time
    total_time = phase1_time + phase2_time
    print(f"⏱️ Phase 2 completed in {phase2_time / 60:.1f} minutes")
    print(f"🎯 Total training time: {total_time / 60:.1f} minutes")

    # =============================================================================
    # EVALUATION ON TEST SET
    # =============================================================================
    print("📊 Evaluating on test set...")

    # Load best model
    model.load_state_dict(torch.load('../results/best_model_final.pth'))
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_acc = 100. * correct / total
    print(f"🎯 Test Accuracy: {test_acc:.2f}%")
    print(f"📉 Test Loss: {test_loss / len(test_loader):.4f}")

    # Classification report
    class_names = test_dataset.classes
    print("\n📋 Classification Report:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))

    # =============================================================================
    # VISUALIZATION
    # =============================================================================
    print("📈 Creating visualizations...")

    # Combine training histories
    train_losses = train_losses_1 + train_losses_2
    train_accs = train_accs_1 + train_accs_2
    val_losses = val_losses_1 + val_losses_2
    val_accs = val_accs_1 + val_accs_2

    # Plot training history
    plt.figure(figsize=(15, 5))

    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(train_accs, label='Train Accuracy', color='blue')
    plt.plot(val_accs, label='Validation Accuracy', color='red')
    plt.axvline(x=epochs_phase1, color='green', linestyle='--', label='Fine-tuning starts')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.axvline(x=epochs_phase1, color='green', linestyle='--', label='Fine-tuning starts')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Confusion matrix
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(all_targets, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    plt.savefig('results/pytorch_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save final model
    torch.save(model.state_dict(), '../results/emotion_model_pytorch.pth')
    print("✅ Model saved!")

    # =============================================================================
    # PERFORMANCE SUMMARY
    # =============================================================================
    print(f"\n🎉 Training Complete!")
    print(f"📊 Final Test Accuracy: {test_acc:.2f}%")
    print(f"⏱️ Total Training Time: {total_time / 60:.1f} minutes")
    print(f"🚀 Expected GPU Training Time: 1-2 hours")
    print(f"🐌 Expected CPU Training Time: 6-10 hours")
    print(f"💾 Model saved as: emotion_model_pytorch.pth")

    # Example usage
    # emotion, confidence = predict_emotion("path/to/image.jpg", model, device, val_test_transform, class_names)
    # print(f"Predicted emotion: {emotion} (confidence: {confidence:.2f})")


if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing
    main()