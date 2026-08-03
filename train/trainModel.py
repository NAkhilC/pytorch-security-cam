
import torch
from config import EPOCHS
from models.faceCNN import FaceCNN


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in train_loader:
        # Debugging line
        print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy


def validate(model, value_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in value_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    epoch_loss = running_loss / len(value_loader.dataset)
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy


EPOCHS = EPOCHS


def train_model(model, train_loader, value_loader, criterion, optimizer, device, num_classes, full_dataset):
    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        value_loss, value_accuracy = validate(
            model, value_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} - "
              f"Validation Loss: {value_loss:.4f}, Validation Accuracy: {value_accuracy:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "index_to_name": full_dataset.index_to_name,
    }, "face_cnn_model.pth")
    print("Saved model to face_cnn_model.pth")


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint["num_classes"]
    index_to_name = checkpoint["index_to_name"]

    model = FaceCNN(num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, index_to_name
