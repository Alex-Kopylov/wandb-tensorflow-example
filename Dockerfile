FROM tensorflow/tensorflow:2.9.1-gpu
RUN pip install --upgrade pip
COPY . /src
WORKDIR /src
RUN pip install -r requirements.txt
CMD python train.py
