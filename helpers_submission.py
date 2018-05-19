#!/usr/bin/env python3

"""
Helper functions for submissions.
"""

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image

def labels_to_img(labels, w=16, h=16, imgwidth=400, imgheight=400):
    """
    Takes an array of labels and create the corresponding image with the right
    patch size and image size.
    """
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def labels_to_images(labels, n=100, w=16, h=16, imgwidth=400, imgheight=400):
    """
    Takes a list of labels and create the corresponding images.
    """
    images = []
    step = int(len(labels)/n)
    for i in range(n):
        images.append(labels_to_img(labels[i*step:(i + 1)*step], w=16, h=16, imgwidth=imgwidth, imgheight=imgheight))
    return np.asarray(images)

def patch_to_label(patch, threshold=0.25):
    """
    Takes a single patch and returns its corresponding label.
    1 : Road
    0 : Ground
    """
    df = np.mean(patch)
    if df > threshold:
        return 1
    else:
        return 0

def image_to_string(image, number):
    """
    Takes a single image and its corresponding number for the submission
    and returns a formatted string used to create the csv file.
    """
    patch_size = 16
    for j in range(0, image.shape[1], patch_size):
        for i in range(0, image.shape[0], patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(number, j, i, label))

def make_submission(predictions, filename):
    """
    Takes an array of predicted images and makes a submission file to the
    specified file name.
    """
    with open(filename, 'w') as f:
        f.write('id,prediction\n')
        count = 1
        for fn in predictions:
            f.writelines('{}\n'.format(s) for s in image_to_string(fn, count))
            count += 1