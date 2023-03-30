from flask import Blueprint, render_template, url_for, jsonify, request, send_from_directory
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
import PIL

View = Blueprint( __name__, "View")

def predict_label(img_path):
    # Read image
    model = load_model("fire_nonfire_detection-all-m1.h5", compile=False)
    
    # Preprocess image
    test_image=image.load_img(img_path,target_size=(224, 224))
    test_image=image.img_to_array(test_image)
    test_image=test_image/255
    test_image=np.expand_dims(test_image,axis=0)
    result=(model.predict(test_image) > 0.5).astype("int32")

    Catagories=['Fire','No Fire']

    return Catagories[int(result)]
   

@View.route('/')
def Home():
    return render_template('index.html')

@View.route('/About')
def About():
    return render_template('About.html')

@View.route('/Features')
def Features():
    return render_template('Features.html')


@View.route("/Prediction", methods=['GET', 'POST'])
def main():
	return render_template("Prediction.html")


@View.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/uploads/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("Prediction.html", prediction = p, img_path = img_path)

