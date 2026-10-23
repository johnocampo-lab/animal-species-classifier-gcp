"""
training/train.py
Fine-tunes MobileNetV3-Small on the animal species dataset using transfer learning.
Tracks training & validation loss/accuracy, saves weights, and exports to ONNX.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import get_dataloaders
from training.model import get_model, export_to_onnx

def train(data_dir: str = "data/processed", epochs: int = 5, batch_size: int = 32, lr: float = 0.001, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Starting Training on device: {device} ---")

    loaders, class_names = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    print(f"Recognizing {len(class_names)} species: {class_names}")

    # Save class mapping for serving
    classes_path = os.path.join(output_dir, "classes.json")
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved class mapping to: {classes_path}")

    model = get_model(num_classes=len(class_names), pretrained=True).to(device)

    # Cross-entropy loss with AdamW optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        epoch_train_loss = train_loss / len(loaders["train"].dataset)
        epoch_train_acc = train_correct / len(loaders["train"].dataset)

        # 2. Validation Phase (No Data Augmentation)
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in loaders["val"]:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        epoch_val_loss = val_loss / len(loaders["val"].dataset)
        epoch_val_acc = val_correct / len(loaders["val"].dataset)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    # Save final model weights
    final_weights_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), final_weights_path)
    print(f"Model weights saved to {final_weights_path}")

    # Export to ONNX for Cloud Run serving
    onnx_path = os.path.join(output_dir, "model.onnx")
    export_to_onnx(model.to("cpu"), onnx_path)
    print("Training and export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train animal classifier")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.output_dir)
