#!/usr/bin/env python
# coding: utf-8

import load_ship_data
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# set seed
np.random.seed(522)
# load data
path = './data/shipsnet.json'
train, test, val = load_ship_data.load_data_train_test_split(path)

train, labels_tr = train
test, labels_te = test
val, labels_val = val

# flatten pixels only
train_rgb = train.reshape(train.shape[0], 3, 6400)
test_rgb = test.reshape(test.shape[0], 3, 6400)
val_rgb = val.reshape(val.shape[0], 3, 6400)

# swap 2nd and 3rd axis so that channels are in last place as expected
train_rgb = np.swapaxes(train_rgb, 1, 2)
test_rgb = np.swapaxes(test_rgb, 1, 2)
val_rgb = np.swapaxes(val_rgb, 1, 2)

def images_to_n_colors(images, n_colors, n_subpixels=1000):
    """
    Takes a set of images in (samples, pixels, channels) format and
    returns n_colors # of 3-channel centers and per-pixel labels
    """
    all_centers, all_labels = [], []
    num_imgs = images.shape[0]
    for i in range(num_imgs):
        kmeans = KMeans(n_clusters=n_colors, random_state=522)
        # divide by 255 so that it will work with imshow
        img = images[i, :] / 255
        img_sample = shuffle(img, random_state=522)[:n_subpixels]
        kmeans.fit(img_sample)
        labels = kmeans.predict(img)
        all_centers.append(kmeans.cluster_centers_)
        all_labels.append(labels)
    return(all_centers, all_labels)
        
def recreate_image(clusters, labels):
    img = np.zeros((80, 80, 3))
    label_idx = 0
    for i in range(80):
        for j in range(80):
            img[i][j] = clusters[labels[label_idx]]
            label_idx += 1
    return img

start_time = time.time()
centers, labels = images_to_n_colors(train_rgb, 2)
print(time.time() - start_time)
