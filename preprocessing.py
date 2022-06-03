import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import yaml

config = yaml.safe_load(open('configs/config.yaml', 'r'))


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def preprocess(dataset, batch_size):
  """Preprocesses the dataset."""
  dataset = dataset.map(normalize_img)
  dataset = dataset.batch(batch_size)
  dataset = dataset.cache()
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  return dataset