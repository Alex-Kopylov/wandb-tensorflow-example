FROM tensorflow/tensorflow:2.9.1-gpu
RUN pip install --upgrade pip
RUN python3 -m pip install pyaml tensorflow_datasets wandb albumentations matplotlib
COPY . /src
WORKDIR /src
CMD python train.py