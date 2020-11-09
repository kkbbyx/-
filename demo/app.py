import numpy as np
from PIL import Image, ImageChops
import base64
import re
import tensorflow.keras as keras
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
model_path='model'
global model
model = keras.models.load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/', methods=['Get', 'POST'])
def predict():
    parseImage(request.get_data())
    img = img_to_array(load_img('output.png', target_size=(28, 28), color_mode="grayscale")) / 255.
    img = np.expand_dims(img, axis=0)
    code = model.predict_classes(img)[0]
    response = int(code)
    #return response
    return jsonify(response)
def parseImage(imgData):
    imgStr = re.search(b'base64,(.*)', imgData).group(1)
    with open('./output.png', 'wb') as output:
        output.write(base64.decodebytes(imgStr))

if __name__ == "__main__":
    app.run()