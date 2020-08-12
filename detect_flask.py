import numpy as np
import argparse
import time
import cv2
import os
import sys
# from flask import Flask, request, Response, jsonify, render_template
from flask import Flask, request, Response, jsonify,send_file, render_template, make_response
from flask_cors import CORS, cross_origin
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
# CORS(app)

# @app.route('/')
# def home():
#     return render_template('home.html')
# # route http posts to this method
@app.route('/api/yoloPredict', methods=['POST'])
@cross_origin()
def yolo_predict():
    data = request.get_data()
    print('huhu', type(data))
   
    img = Image.open(BytesIO(base64.b64decode(data)))
    print(type(img))
    npimg=np.array(img)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    res, text =get_predection(image,nets,Lables,Colors)
    image=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
    np_img=Image.fromarray(image)
    np_img.save('./predict/predict.jpg')
    # np_img = image_to_byte_array(np_img).decode("utf-8")
    # np_img = base64.b64encode(image).decode("assci")
    with open("predict/predict.jpg", "rb") as image_file:
        data = base64.b64encode(image_file.read())
        data = data.decode('utf-8')
    
    print(type(data))
    # with(open('hi.txt', 'w')) as fs:
    #     fs.write(np_img)
    try:
    
        tiem = {'img': data,
                'txt':text
                }
        
        # print(tiem)
        with(open('hi.txt', 'w')) as fs:
            fs.write(data)
        return jsonify(tiem)
    except:
        print('loiiiiiiiiiiiiiiiii!')
        return data
   

    # start flask app
if __name__ == '__main__':
    app.run(debug=True)