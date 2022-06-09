from functools import partial

import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from svc.data_tests import verify_shape


def encode_labels(image, label, n_classes):
    """Encodes the labels."""
    return image, tf.one_hot(label, n_classes)


def resize(image, label, img_shape):
    image = tf.image.resize(image, img_shape)
    return image, label


def reshape_img(image, label, img_shape):
    return tf.reshape(image, [*img_shape, 3]), label


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def preprocess(dataset, img_shape, n_classes):
    """Preprocesses the dataset."""
    dataset = dataset.map(partial(resize, img_shape=img_shape), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(partial(encode_labels, n_classes=n_classes), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(normalize_img, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(partial(reshape_img, img_shape=img_shape), num_parallel_calls=AUTOTUNE)
    return dataset
