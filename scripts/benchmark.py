import time
import torch
from PIL import Image
from torchvision import transforms

from src.model import ScreenDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ScreenDetector().to(DEVICE)
model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

image = Image.open("demo/real1.jpeg").convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# Warm-up
for _ in range(10):
    with torch.no_grad():
        model(image)

start = time.time()

for _ in range(100):
    with torch.no_grad():
        model(image)

end = time.time()

avg_ms = (end-start)/100*1000

print(f"\nAverage inference time : {avg_ms:.2f} ms/image")
print(f"FPS : {1000/avg_ms:.2f}")