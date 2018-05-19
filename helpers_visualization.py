#!/usr/bin/env python3

"""
Helper functions for visualization.
"""

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image

def img_float_to_uint8(img):
    """
    Converts an image to the int8 format.
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    """
    Concatenates two images (for example an image and its groundtruth) to
    visualize both of them easily.
    """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    """
    Creates an image with an overlay to visualize the prediction on
    the original image.
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def visualize_overlay(img, predicted_img, size=8):
    """
    Plots the image overlay.
    """
    new_img = make_img_overlay(img, predicted_img)
    plt.figure(figsize=(size, size))
    plt.imshow(new_img)
    plt.show()
    
def visualize_prediction(img, pred_img):
    """
    Plots an image, its prediction and the overlay.
    """
    cimg = concatenate_images(img, pred_img)
    fig1 = plt.figure(figsize=(15, 15)) # create a figure with the default size 
    plt.imshow(cimg, cmap='Greys_r')
    plt.show()

    visualize_overlay(img, pred_img)