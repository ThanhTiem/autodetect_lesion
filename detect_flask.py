import argparse
import base64
import io as StringIO
import json
import os
import time
from io import BytesIO

import cv2
import numpy as np
# from flask import Flask, request, Response, jsonify, render_template
from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, url_for)
from PIL import Image
from werkzeug.utils import secure_filename

from frcnn.test_frcnn import detect_img
from model.model_yolo import *

confthres = 0.3
nmsthres = 0.1
yolo_path = './'


# labelsPath="obj.names"
# cfgpath="cfg/yolov3_nocrop.cfg"
# wpath="weight/yolov3_nocrop.weights"
# Lables=get_labels(labelsPath)
# CFG=get_config(cfgpath)
# Weights=get_weights(wpath)
# nets=load_model(CFG,Weights)
# Colors=get_colors(Lables)
# Initialize the Flask application

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def frcnn_detect(img_name):
    use_horizontal_flips = False
    use_vertical_flips = False
    rot_90 = False
    im_size = 600
    anchor_box_scales = [64, 128, 256, 512]
    anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
    im_size = 600
    img_channel_mean = [103.939, 116.779, 123.68]
    img_scaling_factor = 1.0
    num_rois = 4
    rpn_stride = 16
    balanced_classes = False
    std_scaling = 4.0
    classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
    rpn_min_overlap = 0.3
    rpn_max_overlap = 0.7
    classifier_min_overlap = 0.1
    classifier_max_overlap = 0.5
    class_mapping = {'MALIGNANT': 0, 'BENIGN': 1, 'bg': 2}

    detected_img = detect_img(img_name)

    return detected_img


@app.route('/')
def home():
    return render_template('home.html')
# route http posts to this method


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            filename = detect_img(file_path)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # file.save(os.path.join("static", "images", filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))

    return


@app.route('/upload<filename>', methods=['GET', 'POST'])
def uploaded_file(filename):
    filename = "../static//images/" + filename
    return render_template("result.html", filename=filename)


@app.route('/api/frcnn_detect', methods=['POST'])
def frcc_predict():
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    img.save("mammogram.jpg")
    name, det = detect_img("mammogram.jpg")
    with open("mammogram.jpg", 'rb') as f:
        string_64 = base64.b64encode(f.read())
    f.close()
    return jsonify(str({
        "name": "mammogram.jpg",
        "base64": string_64,
        "label": det[0],
        "conf": str(det[1])
        }))


@app.route('/api/yolo_predict', methods=['POST'])
def yolo_predict():

    #image = cv2.imread("./test1.jpg")
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    npimg=np.array(img)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    res, res_crop=get_predection(image,nets,Lables,Colors)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # show the output image
    #cv2.imshow("Image", res)
    #cv2.waitKey()
    image=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
    image_crop=cv2.cvtColor(res_crop,cv2.COLOR_BGR2RGB)
    np_img=Image.fromarray(image)
    np_img_crop=Image.fromarray(image_crop)
    img_encoded=image_to_byte_array(np_img)
    img_encoded_crop=image_to_byte_array(np_img_crop)
    return Response(response=[img_encoded_crop], status=200,mimetype="image/jpeg")

    # start flask app

if __name__ == '__main__':
    app.run(debug=True)
