import torch

model_path = "model/digitRecognizer.pth"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

print("Saved model keys:")
print(checkpoint.keys())  # This should show: dict_keys(['model', 'optimizer'])

print("\nModel state_dict keys:")
print(checkpoint['model'].keys())  # This will print the layer names
