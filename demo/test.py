'''import tensorflow.keras as keras
model_path='model'
model = keras.models.load_model(model_path)
model.summary()'''
from flask import Flask, render_template
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True)