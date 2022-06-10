import tensorflow as tf
import tensorflow_datasets
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.data import AUTOTUNE
from wandb.keras import WandbCallback

from svc.augmentation import augment
from svc.data_tests import pass_tests_before_fitting
from svc.model import build_model
from svc.preprocessing import preprocess
from svc.visualization import log_image_artifacts_to_wandb


def train(config):
    tf.random.set_seed(config['random_seed'])

    split_cfg = config['dataset']['split']
    (train_ds, val_ds, test_ds), metadata = tensorflow_datasets.load(
        name=config['dataset']['name'],
        data_dir=config['dataset']['data_dir'],
        split=[split_cfg['train'], split_cfg['val'], split_cfg['test']],
        with_info=True,
        as_supervised=True,
    )

    # visualization
    log_image_artifacts_to_wandb(data=train_ds, metadata=metadata)

    # preprocessing
    train_ds = preprocess(train_ds, img_shape=config['img_shape'], n_classes=config['n_classes'])
    val_ds = preprocess(val_ds, img_shape=config['img_shape'], n_classes=config['n_classes'])
    val_ds = val_ds.batch(config['batch_size'])
    val_ds = val_ds.cache()
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # augment images
    train_ds = augment(data=train_ds)
    train_ds = train_ds.batch(config['batch_size'])
    train_ds = train_ds.cache()
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    # model
    model = build_model(input_shape=config['img_shape'], output_shape=config['n_classes'])
    callbacks = [EarlyStopping(**config['callbacks']['EarlyStopping']),
                 ReduceLROnPlateau(**config['callbacks']['ReduceLROnPlateau']),
                 WandbCallback(**config['callbacks']['WandbCallback'])]

    model.compile(optimizer=config['optimizer'],
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=config['metrics'])

    # data tests (pre-fitting)
    pass_tests_before_fitting(data=train_ds, img_shape=config['img_shape'], n_classes=config['n_classes'])
    pass_tests_before_fitting(data=val_ds, img_shape=config['img_shape'], n_classes=config['n_classes'])

    # training
    history = model.fit(train_ds, epochs=config['epochs'], validation_data=val_ds, callbacks=callbacks)
