from flask import Flask, request, jsonify
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision import models, transforms
import json
import os

app = Flask(__name__)

# Define the model checkpoint URL and download path
CHECKPOINT_URL = "https://drive.usercontent.google.com/download?id=1MaI532ClLzwIwRIBy0inU9dmP0P-A9zm&export=download"
CHECKPOINT_PATH = "vgg_finetuned_custom_data.pth"

# Download model checkpoint if not already present
# if not os.path.exists(CHECKPOINT_PATH):
#     response = requests.get(CHECKPOINT_URL)
#     with open(CHECKPOINT_PATH, 'wb') as f:
#         f.write(response.content)

if not os.path.exists(CHECKPOINT_PATH):
    try:
        response = requests.get(CHECKPOINT_URL)
        with open(CHECKPOINT_PATH, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded successfully: {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Model download failed: {e}")

# Load class names (Modify based on your actual file)
with open('classes.json', 'r') as f:
    class_names = json.load(f)



# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(weights=None)
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, len(class_names))  # Adjust for your classes
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.to(device)
model.eval()

def classify_image(image):
    """Classify the input image."""
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred_class = output.max(1)
        pred_class = str(pred_class.item())
        return class_names.get(pred_class, "Unknown class")

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predicting the class of an image from a URL."""
    data = request.get_json()  # Parse the JSON body
    if 'image_url' not in data:
        return jsonify({'error': 'No image_url provided'}), 400
    
    image_url = data['image_url']
    try:
        # Download the image from the URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Classify the image
        label = classify_image(image)
        
        # Return the classification result
        return jsonify({'class': label})
    
    except Exception as e:
        return jsonify({'error': f"Failed to process image: {str(e)}"}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
