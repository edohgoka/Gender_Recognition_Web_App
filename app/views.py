from flask import render_template, request
import os
import numpy as np
import cv2
from app.face_recognition import faceRecognitionPipeline
import matplotlib.image as matimg

UPLOAD_FOLDER = "static/upload/"

#@app.route("/")
def index():
    return render_template("index.html")

def app():
    return render_template("app.html")

def gender():
    if request.method == "POST":
        f = request.files["image_name"]
        filename = f.filename
        # Save image in the upload folder
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path) # save image into upload folder

        # Get predictions
        pred_img, predictions = faceRecognitionPipeline(path)
        pred_filename = "prediction_image.jpg"
        cv2.imwrite(f"./static/predict/{pred_filename}", pred_img)

        #print(predictions)

        #print("ML model predicted successfully")
        report = []

        for i, obj in enumerate(predictions):
            gray_image = obj["roi"] # grayscale image
            eigen_image = obj["eig_img"].reshape(100, 100) # eigen shape
            gender_name = obj["prediction_name"] # name
            score = round(obj["score"] * 100, 2) # probability score

            # Save grayscale and eigen images to folder
            gray_image_name = f"roi_{i}.jpg"
            eigen_image_name = f"eigen_{i}.jpg"
            matimg.imsave(f"./static/predict/{gray_image_name}", gray_image, cmap = "gray")
            matimg.imsave(f"./static/predict/{eigen_image_name}", eigen_image, cmap = "gray")

            # save report 
            report.append([gray_image_name,
                          eigen_image_name,
                          gender_name,
                          score])
            
        return render_template('gender.html',fileupload=True, report=report) # POST REQUEST

    return render_template("gender.html", fileupload = False) # GET request