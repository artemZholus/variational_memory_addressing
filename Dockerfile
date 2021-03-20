FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN pip install \
    scipy==1.6.1 \
    numpy==1.19.2 \
    tensorboardX==2.1 \
    pandas==1.2.3
