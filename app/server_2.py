import streamlit as st 
from flask import Flask, request, jsonify
import torch 
from torchvision import models, transforms 
from PIL import Image 
import json
import requests
import os

# Enhanced Streamlit UI Configuration
st.set_page_config(
    page_title="Butterfly Species Detector (for Gujarat)", 
    page_icon="🦋", 
    layout="wide"
)


app = Flask(__name__)

# Define the model checkpoint URL (Replace with your direct download link)
# CHECKPOINT_URL = "https://drive.google.com/file/d/1TbVPD2jNiFNe6hclk0Uxnk4FVGyTQhk7/view"
CHECKPOINT_URL = https://drive.usercontent.google.com/download?id=1MaI532ClLzwIwRIBy0inU9dmP0P-A9zm&export=download
# Path to save the downloaded checkpoint
CHECKPOINT_PATH = "vgg_finetuned_custom_data.pth"

# Download the checkpoint if not already present
if not os.path.exists(CHECKPOINT_PATH):
    with st.spinner("Downloading model checkpoint..."):
        response = requests.get(CHECKPOINT_URL)
        with open(CHECKPOINT_PATH, 'wb') as f:
            f.write(response.content)
        st.success("Model checkpoint downloaded!")

# Load class names 
with open('classes.json', 'r') as f: 
    class_names = json.load(f)  # Expecting class_names as a dictionary 

# Define the transformation 
transform = transforms.Compose([ 
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
]) 

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = models.vgg16(weights=None)  # Initialize model without pretrained weights 
num_features = model.classifier[6].in_features 
model.classifier[6] = torch.nn.Linear(num_features, len(class_names))  # Adjust to the number of classes 
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device)) 
model = model.to(device) 
model.eval() 

def google_search_link(scientific_name):
    """Generate a Google search link for the scientific name"""
    search_query = f"https://www.google.com/search?q={scientific_name}+butterfly"
    return search_query

def classify_image(image): 
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension 
    with torch.no_grad(): 
        output = model(input_tensor) 
        _, pred_class = output.max(1) 
        pred_class = str(pred_class.item())  # Convert to string to match JSON keys 
        return class_names.get(pred_class, "Unknown class")  # Retrieve class name, with fallback 

# @app.route('/predict', methods=['POST'])
# def predict():
#     print("in")
#     """API endpoint for predicting the class of an uploaded image."""
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     image = request.files['image']
#     image = Image.open(image).convert('RGB')
#     print("out")
#     label = classify_image(image)
    
#     return jsonify({'class': label})


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

# Main Streamlit App
def main():
    st.title("🦋 Butterfly Species Detector")
    st.write("Upload a butterfly image to classify its species and learn more!")

    # Sidebar
    st.sidebar.header("Model Information")
    st.sidebar.write("This model detects butterfly species based on uploaded images.")

    # Image Upload
    uploaded_file = st.file_uploader("Upload a Butterfly Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner('Analyzing Image...'):
            label = classify_image(image)
        
        st.success(f"Detected Species: **{label}**")
        
        # Google Search Link
        st.markdown(f"[🔍 Google '{label} butterfly']({google_search_link(label)})", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

