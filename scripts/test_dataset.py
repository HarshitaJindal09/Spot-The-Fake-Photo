from src.dataset import get_dataloaders

train_loader, val_loader, test_loader, classes = get_dataloaders()

print("Classes:", classes)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches:", len(test_loader))