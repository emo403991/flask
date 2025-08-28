from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import gdown
import os


app = Flask(__name__)
model_path = "model.h5"
if not os.path.exists(model_path):
    url = "https://drive.google.com/file/d/1ShQ9H1sRmPJeJOgyNO5P5YeYP0H-D79h/view?usp=drive_link" 
    gdown.download(url, model_path, quiet=False)

model = load_model(model_path)

def preprocess_image(image):
    img = image.resize((224, 224)).convert('RGB')
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file.stream)
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)[0][0]
        result = 'مصاب' if prediction > 0.5 else 'طبيعي'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
