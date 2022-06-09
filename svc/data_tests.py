from functools import partial

import tensorflow as tf
from tensorflow.python.data import AUTOTUNE


def verify_shape(image, label, img_shape, n_classes):
    image = tf.ensure_shape(x=image, shape=[None, *img_shape, 3])
    label = tf.ensure_shape(x=label, shape=[None, n_classes])
    return image, label


def pass_tests_before_preprocessing():
    pass


def pass_tests_before_fitting(data, img_shape, n_classes):
    data.map(partial(verify_shape, img_shape=img_shape, n_classes=n_classes), num_parallel_calls=AUTOTUNE)
