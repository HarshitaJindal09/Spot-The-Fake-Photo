import os
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import get_dataloaders
from src.model import ScreenDetector


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    _, _, test_loader, classes = get_dataloaders()

    model = ScreenDetector().to(DEVICE)

    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("\nClassification Report\n")
    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    os.makedirs("results", exist_ok=True)

    plt.savefig("results/confusion_matrix.png")

    plt.show()


if __name__ == "__main__":
    main()