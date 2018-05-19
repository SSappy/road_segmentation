#!/usr/bin/env python3

"""
Helper functions for loading and extraction.
"""

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import tensorflow as tf


def load_image(infilename):
    """
    Loads a single image taking its file name in argument.
    """
    return mpimg.imread(infilename)

def load_training_set(root_dir='training/'):
    """
    Loads the whole training set (both images and groud truth) provided that the root
    directory and the image directory are the ones specified.
    """
    image_dir = root_dir + 'images/'
    files = os.listdir(image_dir)
    imgs = [load_image(image_dir + files[i]) for i in range(len(files))]
    
    gt_dir = root_dir + "groundtruth/"
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(len(files))]
    
    return np.asarray(imgs), np.asarray(gt_imgs)

def load_testing_set(root_dir='test_set_images/'):
    """
    Loads the whole testing set.
    """
    files = sorted(os.listdir(root_dir))
    imgs = []
    for fdir in files:
        file = os.listdir(root_dir + fdir + '/')
        imgs.append(load_image(root_dir + fdir + '/' + file[0]))

    return np.asarray(imgs)
    
def img_crop(im, w=16, h=16, stride=16, padding=0):
    """
    Crop an image into patches of w*h size, with the specified stride.
    A padding can be added for larger window sizes than 16.
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    if padding != 0:
        im = np.lib.pad(im, ((padding, padding), (padding, padding), (0,0)), 'reflect')
    for i in range(padding,imgheight+padding,stride):
        for j in range(padding,imgwidth+padding,stride):
            if is_2d:
                im_patch = im[j-padding:j+w+padding, i-padding:i+h+padding]
            else:
                im_patch = im[j-padding:j+w+padding, i-padding:i+h+padding, :]
            list_patches.append(im_patch)
    return list_patches

def extract_patches(imgs, w=16, h=16, stride=16, padding=0):
    """
    Extract all the patches from an array of images with the specified
    size, stride and padding.
    """
    img_patches = [img_crop(imgs[i], w=w, h=h, stride=stride, padding=padding) for i in range(len(imgs))]

    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

    return img_patches

def pad_image(data, padding):
    """
    Pad an image with mirror boundary conditions for window sizes larger than 16.
    """
    #Black&white images
    if len(data.shape) < 3:
        data = np.lib.pad(data, ((padding, padding), (padding, padding)), 'reflect')
    #Colored image (RGB)
    else:
        data = np.lib.pad(data, ((padding, padding), (padding, padding), (0,0)), 'reflect')
    return data

def value_to_class(v, threshold=0.25):
    """
    Convert a ground truth patch to its corresponding class.
    1 : Road
    0 : Ground
    """
    df = np.sum(v)
    if df > threshold:
        return 1
    else:
        return 0

def patches_to_labels(gt_patches, threshold=0.25):
    """
    Convert an array of (extracted) patches to their corresponding labels (classes).
    1 : Road
    0 : Ground
    """
    return np.asarray([value_to_class(np.mean(gt_patches[i]), threshold) for i in range(len(gt_patches))])

def patches_to_features(img_patches, feature_extraction_function):
    """
    Function used for the baseline models where the features have to be built by
    ourselves. Takes an array of (extracted) patches and extract meaningful features
    from it using the feature_extraction_function parameter (a function extracting
    the features).
    """
    return np.asarray([feature_extraction_function(img_patches[i]) for i in range(len(img_patches))])

def group_patches(patches, num_images):
    """
    Regroup extracted patches to make meaningful images.
    """
    return patches.reshape(num_images, -1)