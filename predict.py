import sys
import torch
from PIL import Image
from torchvision import transforms

from src.model import ScreenDetector


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = ScreenDetector().to(DEVICE)

    model.load_state_dict(
        torch.load(
            "models/best_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(image_path):

    model = load_model()

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        # Probability of Screen Photo (Class = 1)
        score = probabilities[0][1].item()

    print(f"{score:.4f}")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python predict.py image.jpg")
        sys.exit(1)

    predict(sys.argv[1])