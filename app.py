from flask import Flask, render_template, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(2048, 7)  # Adjust for number of emotion categories
model.load_state_dict(torch.load('custom_resnet50_emotion_model .pth'))
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Emotion classes (adjust based on your dataset)
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Preprocess the image
    img = Image.open(io.BytesIO(file.read()))
    img = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Predict emotion
    with torch.no_grad():
        output = model(img)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = probs.max(1)
    
    # Prepare result
    emotion = emotions[predicted_class.item()]
    percentage = confidence.item() * 100
    
    return jsonify({'emotion': emotion, 'confidence': f'{percentage:.2f}%'})

if __name__ == '__main__':
    app.run(debug=True)
