import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.model import ScreenDetector

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Spot the Fake Photo",
    page_icon="📷",
    layout="centered"
)

# -------------------------------------------------
# Device
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
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

model = load_model()

# -------------------------------------------------
# Classes
# -------------------------------------------------
classes = ["Real Photo", "Screen Photo"]

# -------------------------------------------------
# Image Transform
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
### Spot the Fake Photo

**Framework:** PyTorch

**Model:** ResNet18 (Transfer Learning)

**Dataset:** Real vs Screen Images

**Validation Accuracy:** **100%***

**Inference Time:** **31.70 ms/image**

**FPS:** **31.54**

\*Validation accuracy was achieved on the provided dataset.
""")

# -------------------------------------------------
# Main Title
# -------------------------------------------------
st.title("📷 Spot the Fake Photo")

st.write(
    """
Upload an image and the model will determine whether it is a **Real Photo**
or a **Photo Taken of a Screen**.

**Prediction Score Interpretation**

- **0.0 → Real Photo**
- **1.0 → Screen Photo**
"""
)

st.divider()

# -------------------------------------------------
# Upload Image
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)

    # Probability of Screen Photo (Class 1)
    score = probabilities[0][1].item()

    if score >= 0.5:
        prediction = "SCREEN PHOTO"
        st.error("📱 Prediction: SCREEN PHOTO")
    else:
        prediction = "REAL PHOTO"
        st.success("✅ Prediction: REAL PHOTO")

    st.divider()

    # -------------------------------------------------
    # Prediction Score
    # -------------------------------------------------
    st.metric(
        label="Prediction Score (0–1)",
        value=f"{score:.4f}"
    )

    st.progress(score)

    st.info(
        f"""
**Prediction Score:** **{score:.4f}**

Interpretation:

- **0.0 → Real Photo**
- **1.0 → Screen Photo**

The model predicts **{prediction}** based on this score.
"""
    )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()

st.caption(
    "Developed using PyTorch • ResNet18 • Streamlit • Computer Vision"
)