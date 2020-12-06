import load_ship_data
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# set seed
np.random.seed(522)

def kmeans_image_to_ncolors(img, n_colors, n_subpixels=1000):
    kmeans = KMeans(n_clusters=n_colors, random_state=522)
    img = img / 255
    img_sample = shuffle(img, random_state=522)[:n_subpixels]
    kmeans.fit(img_sample)
    labels = kmeans.predict(img)
    return(kmeans.cluster_centers_, labels)
        
def recreate_image(centers, labels, n_colors):
    img = np.zeros((80, 80, 3))
    label_idx = 0
    for i in range(80):
        for j in range(80):
            img[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return img.reshape(19200)

def return_flattened_ncolor_images(samples, n_colors):
    n_samples = samples.shape[0]
    samples_flat_ncolors = np.zeros((n_samples, 19200))
    for i in range(n_samples):
        centers, labels = kmeans_image_to_ncolors(samples[i, :], n_colors)
        samples_flat_ncolors[i, :] = recreate_image(centers, labels, n_colors)
    return samples_flat_ncolors

def load_data_and_convert(path, n_colors):
    train, test, val = load_ship_data.load_data_train_test_split(path)
    # split labels
    train, labels_tr = train
    test, labels_te = test
    val, labels_val = val
    # combine train and val
    train = np.concatenate((train, val), axis=0)
    labels_tr = np.concatenate((labels_tr, labels_val), axis=None)
    # flatten pixels only
    train_rgb = train.reshape(train.shape[0], 3, 6400)
    test_rgb = test.reshape(test.shape[0], 3, 6400)
    train_flat = train.reshape(train.shape[0], 19200)
    print(train_flat[0][:10])
    # move channels to last dim
    train_rgb = np.swapaxes(train_rgb, 1, 2)
    test_rgb = np.swapaxes(test_rgb, 1, 2)
    # reduce to n images and flatten
    train_flat_2colors = return_flattened_ncolor_images(train_rgb, 2)
    test_flat_2colors = return_flattened_ncolor_images(test_rgb, 2)
    return train_flat_2colors, test_flat_2colors, labels_tr, labels_te

# example use:
#path = './data/shipsnet.json'
#n_colors = 2
#train, test, labels_tr, labels_te = load_data_and_convert(path, n_colors)
