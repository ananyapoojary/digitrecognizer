import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# ✅ Load the trained model
class DigitRecognizerCNN(nn.Module):
    def __init__(self):
        super(DigitRecognizerCNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.main(x)

# Load model
model = DigitRecognizerCNN()
model_path = "model/digitRecognizer.pth"

checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'], strict=False)  # ✅ Ignore extra/missing keys

model.eval()
print("Model loaded successfully!")

# ✅ Define image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image_path):
    """Load and preprocess an image"""
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict_digit(image_path):
    """Predict digit from an image"""
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    return prediction

# ✅ Run the prediction on a test image
image_path = "images/test_digit.png"  # Change if needed
predicted_digit = predict_digit(image_path)
print(f"🧠 Predicted Digit: {predicted_digit}")

# ✅ Display the image with the predicted digit
img = Image.open(image_path)
plt.imshow(img, cmap="gray")
plt.title(f"Predicted Digit: {predicted_digit}")
plt.show()
