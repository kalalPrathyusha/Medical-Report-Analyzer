# scan_type_predictor.py

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# --- Load model checkpoint ---
checkpoint_path = "scan_type_classifier.pth"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Extract class labels
class_names = checkpoint['class_names']

# --- Define model architecture ---
model = resnet18(weights=None)  # Avoid ImageNet weights since we're loading ours
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Match output layer size
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Define image transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Prediction function ---
def predict_scan_type(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).squeeze()
        pred_idx = torch.argmax(probs).item()
        pred_label = class_names[pred_idx]
        confidence = probs[pred_idx].item()
    return pred_label, confidence

# --- Manual test block ---
if __name__ == "__main__":
    test_img_path = r"C:\\Users\\Prath\\OneDrive\\Desktop\\FinalProject\\M2test\\8e49879424380b75830066020b6d35_jumbo.jpg"
    img = Image.open(test_img_path).convert("RGB")
    pred_label, confidence = predict_scan_type(img)
    print(" Predicted Scan Type:", pred_label)
    print(" Confidence Score:", f"{confidence:.2f}")



