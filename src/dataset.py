import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(
    dataset_path="dataset",
    batch_size=16,
    image_size=224
):

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, "val"),
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, "test"),
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    class_names = train_dataset.classes

    return (
        train_loader,
        val_loader,
        test_loader,
        class_names
    )