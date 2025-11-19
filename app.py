import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import pickle

# ------------------- Load class names -------------------
with open("classes.pkl", "rb") as f:
    classes = pickle.load(f)

# ------------------- Model Definition (MUST match training) -------------------
class RetinaCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,3,padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2))
        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2))
        self.conv_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128,3,padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(), torch.nn.MaxPool2d(2))
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(128,128)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(128,64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc3 = torch.nn.Linear(64,4)

    def forward(self, x):
        x = self.conv_block1(x); x = self.conv_block2(x); x = self.conv_block3(x)
        x = self.adaptive_pool(x); x = torch.flatten(x,1)
        x = torch.relu(self.fc1(x)); x = self.dropout1(x)
        x = torch.relu(self.fc2(x)); x = self.dropout2(x)
        return self.fc3(x)

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    model = RetinaCNN()
    model.load_state_dict(torch.load("model_epoch_50.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ------------------- Transform -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="OCT Retina Classifier", layout="centered")
st.title("Retinal OCT Disease Classifier")
st.write("Upload an OCT scan → Get instant prediction (CNV, DME, DRUSEN, NORMAL)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded OCT Scan", width="stretch")

    # Predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)
        predicted_class = classes[pred_idx.item()]

    st.markdown(f"### **Prediction: {predicted_class}**")
    st.progress(conf.item())
    st.write(f"**Confidence: {conf.item():.2%}**")

    if predicted_class == "NORMAL":
        st.balloons()
        st.success("Healthy Retina Detected!")
    else:
        st.warning("Possible pathology detected — consult an ophthalmologist.")