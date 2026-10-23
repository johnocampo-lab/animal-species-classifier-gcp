"""
training/evaluate.py
Evaluates the fine-tuned model strictly on the untouched test split.
Generates:
1. Classification report (Accuracy, Precision, Recall, F1-Score per animal species)
2. Confusion matrix heatmap saved as evaluation_results/confusion_matrix.png
3. Metrics summary JSON saved as evaluation_results/metrics.json
"""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data.dataset import get_dataloaders
from training.model import get_model

def evaluate(data_dir: str = "data/processed", model_dir: str = "models", output_dir: str = "evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)
    classes_path = os.path.join(model_dir, "classes.json")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Missing {classes_path}. Train the model first.")

    with open(classes_path) as f:
        class_names = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Evaluating model on held-out test split (Device: {device}) ---")

    loaders, _ = get_dataloaders(data_dir=data_dir, batch_size=32)
    model = get_model(num_classes=len(class_names), pretrained=False)
    
    weights_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loaders["test"]:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=class_names)

    print("\n================ Classification Report ================")
    print(report_text)
    print(f"Overall Test Accuracy: {acc * 100:.2f}%\n")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved metrics summary to: {metrics_path}")

    # Generate Confusion Matrix Heatmap
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Species", fontsize=12)
    plt.ylabel("True Species", fontsize=12)
    plt.title(f"Animal Species Recognition - Confusion Matrix (Test Acc: {acc*100:.1f}%)", fontsize=13)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved Confusion Matrix visualization to: {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate animal species classifier")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--output-dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    evaluate(args.data_dir, args.model_dir, args.output_dir)
