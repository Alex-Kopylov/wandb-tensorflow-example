import keras_cv
from tensorflow.data.experimental import AUTOTUNE


def aug_fn(image, label):
    rand_augment = keras_cv.layers.RandAugment(
        value_range=(0, 255),
        augmentations_per_image=3,
        magnitude=0.3,
        magnitude_stddev=0.2,
        rate=0.5,
    )
    image = rand_augment(image)
    return image, label


def augment(data):
    return data.map(aug_fn, num_parallel_calls=AUTOTUNE)
