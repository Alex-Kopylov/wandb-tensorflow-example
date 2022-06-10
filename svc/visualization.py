import tensorflow_datasets as tfds
import wandb


def log_image_artifacts_to_wandb(data, metadata):
    fig = tfds.show_examples(data, metadata, cols=6, rows=8)
    wandb.log({"plot": fig})
