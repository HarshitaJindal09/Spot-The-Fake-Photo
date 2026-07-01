import os
import random
import shutil

# -----------------------------
# Dataset Path
# -----------------------------
dataset_path = "dataset"

real_path = os.path.join(dataset_path, "real")
screen_path = os.path.join(dataset_path, "screen")

# Output folders
train_real = os.path.join(dataset_path, "train", "real")
train_screen = os.path.join(dataset_path, "train", "screen")

val_real = os.path.join(dataset_path, "val", "real")
val_screen = os.path.join(dataset_path, "val", "screen")

test_real = os.path.join(dataset_path, "test", "real")
test_screen = os.path.join(dataset_path, "test", "screen")


# Create folders
folders = [
    train_real, train_screen,
    val_real, val_screen,
    test_real, test_screen
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)


def split_images(source_folder, train_folder, val_folder, test_folder):
    images = os.listdir(source_folder)

    random.shuffle(images)

    total = len(images)

    train_size = int(0.7 * total)
    val_size = int(0.15 * total)

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    for image in train_images:
        shutil.copy(
            os.path.join(source_folder, image),
            os.path.join(train_folder, image)
        )

    for image in val_images:
        shutil.copy(
            os.path.join(source_folder, image),
            os.path.join(val_folder, image)
        )

    for image in test_images:
        shutil.copy(
            os.path.join(source_folder, image),
            os.path.join(test_folder, image)
        )

    print(f"{source_folder}")
    print(f"Train : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Test : {len(test_images)}")
    print()


split_images(real_path, train_real, val_real, test_real)
split_images(screen_path, train_screen, val_screen, test_screen)

print("Dataset Successfully Split!")