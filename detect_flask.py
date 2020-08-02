import numpy as np
import argparse
import time
import cv2
import os
from flask import Flask, request, Response, jsonify, render_template
import jsonpickle
#import binascii
import io as StringIO
import base64
from io import BytesIO
import io
import json
from PIL import Image
from model.model_yolo import *

confthres = 0.3
nmsthres = 0.1
yolo_path = './'


labelsPath="obj.names"
cfgpath="cfg/yolov3_nocrop.cfg"
wpath="weight/yolov3_nocrop.weights"
Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
nets=load_model(CFG,Weights)
Colors=get_colors(Lables)
# Initialize the Flask application
app = Flask(__name__)
@app.route('/home')
def home():
    return render_template('home.html')
# route http posts to this method
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