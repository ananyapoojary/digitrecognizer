import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# ✅ Define the CNN Model
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
            nn.Linear(128, 10)  # 10 output classes (digits 0-9)
        )

    def forward(self, x):
        return self.main(x)

# ✅ Load Model
model = DigitRecognizerCNN()
model_path = "model/digitRecognizer.pth"

# Ensure model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file '{model_path}' not found!")

checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
print("✅ Model loaded successfully!")

# ✅ Define Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_digit(digit_img):
    """Resize and normalize a single digit image before passing to CNN"""
    digit_img = Image.fromarray(digit_img)  # Convert to PIL image
    digit_img = transform(digit_img)  # Apply transformations
    digit_img = digit_img.unsqueeze(0)  # Add batch dimension
    return digit_img

def predict_digit(digit_img):
    """Classify a single digit"""
    with torch.no_grad():
        output = model(digit_img)
        prediction = torch.argmax(output, dim=1).item()
    return prediction

# ✅ Detect and Segment Digits
def segment_and_classify(image_path):
    """Detect, extract, and classify multiple digits in an image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file '{image_path}' not found!")
    
    print(f"✅ Image file '{image_path}' found.")

    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply adaptive thresholding for better digit detection
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)

    # Find contours (possible digit regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_predictions = []  # Store digit predictions
    bounding_boxes = []  # Store bounding boxes for sorting

    # Loop through detected contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore small regions (noise) and non-digit-like shapes
        if w < 10 or h < 10:
            continue

        # Extract the digit region
        digit = thresh[y:y+h, x:x+w]

        # Resize to 28x28 for the CNN model
        digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

        # Predict digit
        processed_digit = preprocess_digit(digit_resized)
        predicted_digit = predict_digit(processed_digit)

        # Store predictions and bounding boxes
        digit_predictions.append((x, predicted_digit))
        bounding_boxes.append((x, y, w, h))

    # Sort digits by x-coordinate (left to right)
    digit_predictions.sort()

    # ✅ Display Results
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h), (_, digit) in zip(bounding_boxes, digit_predictions):
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_image, str(digit), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save and show the result
    output_path = "output.png"
    cv2.imwrite(output_path, output_image)
    print(f"📸 Output saved as '{output_path}'")

    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted Digits: {''.join(str(d) for _, d in digit_predictions)}")
    plt.axis("off")
    plt.show()

    return ''.join(str(d) for _, d in digit_predictions)

# ✅ Run Prediction
image_path = "images/multi_digit4.png"  # Change to your multi-digit image
predicted_digits = segment_and_classify(image_path)
print(f"🔢 Predicted Number: {predicted_digits}")
