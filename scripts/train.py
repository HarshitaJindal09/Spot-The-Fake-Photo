import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.model import ScreenDetector
from src.trainer import train_one_epoch, validate


def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    train_loader, val_loader, test_loader, classes = get_dataloaders()

    model = ScreenDetector().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    EPOCHS = 10

    best_accuracy = 0
    
    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")

        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.2f}%")

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            torch.save(
                model.state_dict(),
                "models/best_model.pth"
            )

            print("✅ Best model saved!")

    print("\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    os.makedirs("results", exist_ok=True)

    # Loss Curve
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/loss_curve.png")
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(8,5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/accuracy_curve.png")
    plt.close()

    print("\nGraphs saved successfully in results folder.")


if __name__ == "__main__":
    main()