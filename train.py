import wandb

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from wandb.keras import WandbCallback

import yaml

from visualization import log_image_artifacts_to_wandb
from model import build_model
from preprocessing import preprocess


def train(config):

    (train_img, val_img, test_img), metadata = tfds.load(
        config['dataset'],
        data_dir = config['data_dir'],
        split=['train[:80%]', 'train[80%:]', 'test'],
        with_info=True,
        as_supervised=True,
        shuffle_files=True
    )

    # visualization
    log_image_artifacts_to_wandb(data=train_img, metadata=metadata)

    # preprocessing
    train_img = preprocess(train_img, config['batch_size'])
    val_img = preprocess(val_img, config['batch_size'])

    # model
    model = build_model(input_shape=config['img_size'], output_shape=config['n_classes'])
    callbacks = [EarlyStopping(**config['callbacks']['EarlyStopping']),
                 ReduceLROnPlateau(**config['callbacks']['ReduceLROnPlateau']),
                 WandbCallback(**config['callbacks']['WandbCallback'])]
    model.compile(optimizer=config['optimizer'],
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=config['metrics'])
    # training
    history = model.fit(train_img, epochs=config['epochs'],
                        validation_data=val_img,
                        callbacks=callbacks)

if __name__ == '__main__':
    tf.random.set_seed(42)

    wandb.login(key=open('secrets/wandb_key.txt', 'r').read())
    config = yaml.safe_load(open('configs/config.yaml', 'r'))
    wandb.init(project=config['wandb']['project'],
               name=config['wandb']['name'],
               config=config)

    train(config)

    wandb.finish()