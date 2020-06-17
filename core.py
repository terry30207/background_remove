import os
import cv2
import math
import numpy as np
import tensorflow as tf
from utils import load_graph_model, get_input_tensors, get_output_tensors



def image_preprocess(image,targetW,targetH):
    image=cv2.resize(image,(targetW,targetH))
    image=cv2.line(image,(targetW-50,0),(targetW-50,targetH-1),(0,0,0),99)
    x=image.astype(np.float32)
    return image

# make background picture into correct size 
def bg_preprocess(bgPath,targetW,targetH):
    background=cv2.imread(bgPath)
    bg=cv2.resize(background,(targetW,targetH))
    return bg

def make_segment(frame,graph):
    x=frame.astype(np.float32)
    x=x/127.5-1
    sample_image = x[tf.newaxis, ...]
    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor_names = get_input_tensors(graph)
        output_tensor_names = get_output_tensors(graph)
        input_tensor = graph.get_tensor_by_name(input_tensor_names[0])
        results = sess.run(output_tensor_names, feed_dict={input_tensor: sample_image})
    for idx, name in enumerate(output_tensor_names):
        if 'float_segments' in name:
            segments = np.squeeze(results[idx], 0)
# Segmentation MASk
    segmentation_threshold = 0.35
    segmentScores = tf.sigmoid(segments)
    mask = tf.math.greater(segmentScores, tf.constant(segmentation_threshold))
    segmentationMask = tf.dtypes.cast(mask, tf.int32)
    segmentationMask = np.reshape(segmentationMask, (segmentationMask.shape[0], segmentationMask.shape[1]))
    return segmentationMask

def make_mask(segmentationMask,imageW, imageH, stride):
    mask_img=np.zeros([ (int(imageH) // stride)  + 1, (int(imageW) // stride)  + 1, 3])
    mask_img[:,:,0]+=segmentationMask*255
    mask_img[:,:,1]+=segmentationMask*255
    mask_img[:,:,2]+=segmentationMask*255
    mask_img = tf.keras.preprocessing.image.img_to_array(mask_img, dtype=np.uint8)
    return mask_img

def denoise(mask_img):
    mask_img = cv2.medianBlur(mask_img,7)
    mask_img = cv2.medianBlur(mask_img,5)
    mask_img = cv2.medianBlur(mask_img,7)
    mask_img = cv2.medianBlur(mask_img,7)
    return mask_img

def anti_aliasing(mask_img):
    mask_img = cv2.blur(mask_img,(5,10))
    return mask_img

def resize_mask(mask_img,targetW,targetH):
    mask_img=cv2.resize(mask_img,(targetW,targetH),cv2.INTER_LINEAR)
    return mask_img

def binarization(mask_img):
    ret,mask_img = cv2.threshold(mask_img,100,255,cv2.THRESH_BINARY) 
    return mask_img

def make_maskedBg(mask_img, bg):
    mask_iv = np.bitwise_not(np.array(mask_img))
    masked_bg = np.bitwise_and(np.array(bg), np.array(mask_iv))
    return masked_bg

def make_final(graph, image, imageW, imageH, targetW, targetH, stride , bgPath = None, useBackground = False):
    image = image_preprocess(image, targetW, targetH)
    key = cv2.waitKey(1)
    segmentationMask = make_segment(image,graph)
    mask_img = make_mask(segmentationMask,imageW, imageH, stride)
    key = cv2.waitKey(1)
    mask_img = denoise(mask_img)
    mask_img = anti_aliasing(mask_img)
    mask_img = resize_mask(mask_img,targetW,targetH)
    mask_img = binarization(mask_img)
    final = np.bitwise_and(np.array(image), np.array(mask_img))
    if useBackground == False:
        return final
    else:
        bg = bg_preprocess(bgPath,targetW, targetH)
        masked_bg = make_maskedBg(mask_img, bg)
        final = masked_bg+final
        return final
