
# vgg19 model

# import os 
# import numpy as np 
# from PIL import Image 
# import cv2
# from flask import Flask, request, render_template 
# from werkzeug.utils import secure_filename 

# from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.vgg19 import VGG19

# # ========================== Build Model

# base_model = VGG19(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
# x = base_model.output
# flat = Flatten()(x)
# class_1 = Dense(4608, activation='relu')(flat)
# dropout = Dropout(0.2)(class_1)
# class_2 = Dense(1152, activation='relu')(dropout)
# output = Dense(2, activation='softmax')(class_2)

# model_03 = Model(base_model.input, output)
# model_03.load_weights('vgg19_model_01.h5')  # Make sure weights match the above architecture

# # ========================== Flask App Setup

# app = Flask(__name__)

# UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# print("✅ Model loaded. App running at http://127.0.0.1:5000")

# # ========================== Utility Functions

# def get_className(classNo):
#     return "Normal" if classNo == 0 else "Pneumonia"

# def getResult(img_path):
#     # Read and preprocess image
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(img)
#     img = img.resize((128, 128))  # Match model input size
#     img = np.array(img) / 255.0
#     input_img = np.expand_dims(img, axis=0)

#     prediction = model_03.predict(input_img)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     return predicted_class

# # ========================== Routes

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return "No file part"

#     f = request.files['file']
#     if f.filename == '':
#         return "No selected file"

#     try:
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
#         f.save(file_path)
#         value = getResult(file_path)
#         result = get_className(value)
#         return render_template('index.html', prediction=result)

#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return "Something went wrong during prediction."

# # ========================== Run App

# if __name__ == '__main__':
#     app.run(debug=True)

# simple model

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)
model = load_model("simple_cnn_pneumonia.h5")
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

labels = ['Pneumonia', 'Normal']  # Binary classifier

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    file_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            img = cv2.imread(file_path)
            img = cv2.resize(img, (128, 128))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)[0][0]
            prediction = labels[1] if pred < 0.5 else labels[0]  # 0=Normal, 1=Pneumonia

    return render_template('index.html', prediction=prediction, file_path=file_path)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run() # removed debug option
    
