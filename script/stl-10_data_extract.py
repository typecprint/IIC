from __future__ import print_function

import sys
import os, sys, errno
import numpy as np
import matplotlib.pyplot as plt

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urllib

try:
    from imageio import imsave
except:
    from scipy.misc import imsave

print(sys.version_info)

HEIGHT = 96
WIDTH = 96
DEPTH = 3
SIZE = HEIGHT * WIDTH * DEPTH

DATA_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"


# DATA_PATH = os.path.join(DATA_DIR, "stl10_binary/unlabeled_X.bin")


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, "rb") as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, "rb") as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()


def save_image(image, name):
    imsave(name, image, format="png")


def save_unlabeled_images(data_dir, data_path):
    print("Saving images to disk")
    for i, _data_path in enumerate(data_path, start=1):
        index = i // 10000
        directory = os.path.join(data_dir, "unlabeled", str(index))
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = os.path.join(directory, "unlabeled_%06d.png" % i)
        print(filename)
        save_image(_data_path, filename)


def save_labeled_images(data_dir, data_path, label_path, save_dir):
    print("Saving images to disk")
    for i, (_data_path, _label_path) in enumerate(zip(data_path, label_path), start=1):
        directory = os.path.join(data_dir, save_dir, str(_label_path))
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = os.path.join(directory, "%s_%06d.png" % (save_dir, i))
        print(filename)
        save_image(_data_path, filename)


if __name__ == "__main__":
    data_dir = "/Users/yohei/workspace/dataset/STL-10/"
    data_type = "unlabeled"
    data_path = os.path.join(data_dir, f"stl10_binary/{data_type}_X.bin")
    label_path = os.path.join(data_dir, f"stl10_binary/{data_type}_y.bin")
    # test to check if the image is read correctly
    with open(data_path) as f:
        image = read_single_image(f)
        plot_image(image)

    # test to check if the whole dataset is read correctly
    images = read_all_images(data_path)
    print(images.shape)

    if data_type != "unlabeled":
        labels = read_labels(label_path)
        save_labeled_images(data_dir, images, labels, save_dir=data_type)

    else:
        save_unlabeled_images(data_dir, images)
