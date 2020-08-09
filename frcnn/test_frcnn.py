from __future__ import division

import os
import pickle
import sys
import time

import cv2
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model

import resnet as nn
import roi_helpers

sys.setrecursionlimit(40000)

# turn off any data augmentation at test time
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

def format_img_size(img, im_size):
    """ formats the image size based on config """
    img_min_side = float(im_size)
    (height,width,_) = img.shape

    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio

def format_img_channels(img, img_channel_mean, img_scaling_factor):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= img_channel_mean[0]
    img[:, :, 1] -= img_channel_mean[1]
    img[:, :, 2] -= img_channel_mean[2]
    img /= img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img, im_size, img_channel_mean, img_scaling_factor):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, im_size)
    img = format_img_channels(img, img_channel_mean, img_scaling_factor)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)


if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

num_features = 1024

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_path = "model_final_1.hdf5"
print('Loading weights from {}'.format(model_path))
model_rpn.load_weights(model_path, by_name=True)
model_classifier.load_weights(model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True


def detect_img(img_name, num_rois,
               rpn_max_overlap,
               rpn_min_overlap,
               classifier_max_overlap,
               classifier_min_overlap,
               im_size, rpn_stride,
               anchor_box_ratios, anchor_box_scales,
               std_scaling, class_mapping,
               img_channel_mean, img_scaling_factor
               ):

    print(img_name)
    st = time.time()
    # filepath = os.path.join(img_path,img_name)

    img = cv2.imread(img_name)

    X, ratio = format_img(img, im_size, img_channel_mean, img_scaling_factor)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)


    R = roi_helpers.rpn_to_roi(Y1, Y2,
                               anchor_box_scales,
                               anchor_box_ratios,
                               std_scaling,
                               rpn_stride,
                               K.image_dim_ordering(), overlap_thresh=0.5)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0]//num_rois + 1):
        ROIs = np.expand_dims(R[num_rois*jk:num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0]//num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= classifier_regr_std[0]
                ty /= classifier_regr_std[1]
                tw /= classifier_regr_std[2]
                th /= classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([rpn_stride*x, rpn_stride*y, rpn_stride*(x+w), rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))

            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (real_x1, real_y1-0)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    # print('Elapsed time = {}'.format(time.time() - st))
    print(all_dets)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.imwrite('../static/images/{}.png'.format(img_name), img)
# print("tp: {} \nfp: {}".format(tp, fp))

img_name = r"D:\autodetect_lesion\P_00005_RIGHT_CC_FULL.jpg"

detect_img(img_name, num_rois,rpn_max_overlap,
        rpn_min_overlap,
        classifier_max_overlap,
        classifier_min_overlap,
        im_size, rpn_stride,
        anchor_box_ratios, anchor_box_scales,
        std_scaling, class_mapping,
        img_channel_mean, img_scaling_factor
        )