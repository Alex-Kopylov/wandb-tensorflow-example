from functools import partial

import wandb

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.data import AUTOTUNE

from wandb.keras import WandbCallback

import yaml

from svc.augmentation import augment
from svc.data_tests import pass_tests_before_fitting
from svc.visualization import log_image_artifacts_to_wandb
from svc.model import build_model
from svc.preprocessing import preprocess


def train(config):
    tf.random.set_seed(config['random_seed'])

    (train_img, val_img, test_img), metadata = tfds.load(
        name=config['dataset']['name'],
        data_dir=config['dataset']['data_dir'],
        split=[config['dataset']['split']['train'],
               config['dataset']['split']['val'],
               config['dataset']['split']['test']],
        with_info=True,
        as_supervised=True,
    )

    # visualization
    log_image_artifacts_to_wandb(data=train_img, metadata=metadata)

    # preprocessing
    train_img = preprocess(train_img, img_shape=config['img_shape'],
                           n_classes=config['n_classes'])
    train_img = train_img.batch(config['batch_size'])
    train_img = train_img.cache()
    train_img = train_img.prefetch(buffer_size=AUTOTUNE)
    val_img = preprocess(val_img, img_shape=config['img_shape'],
                         n_classes=config['n_classes'])
    val_img = val_img.batch(config['batch_size'])
    val_img = val_img.cache()
    val_img = val_img.prefetch(buffer_size=AUTOTUNE)

    # augment images
    # aug_img = augment(data=train_img, img_shape=config['img_shape'], batch_size=config['batch_size'])
    # train_img = train_img.concatenate(aug_img)
    train_len = train_img.cardinality().numpy()

    # model
    model = build_model(input_shape=config['img_shape'], output_shape=config['n_classes'])
    callbacks = [EarlyStopping(**config['callbacks']['EarlyStopping']),
                 ReduceLROnPlateau(**config['callbacks']['ReduceLROnPlateau']),
                 WandbCallback(**config['callbacks']['WandbCallback'])]

    model.compile(optimizer=config['optimizer'],
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=config['metrics'])
    # tests
    pass_tests_before_fitting(data=train_img, img_shape=config['img_shape'], n_classes=config['n_classes'])
    pass_tests_before_fitting(data=val_img, img_shape=config['img_shape'], n_classes=config['n_classes'])
    # training
    history = model.fit(train_img, epochs=config['epochs'],
                        validation_data=val_img,
                        callbacks=callbacks)
