from src.model import ScreenDetector
import torch

model = ScreenDetector()

x = torch.randn(4, 3, 224, 224)

y = model(x)

print("Output Shape:", y.shape)