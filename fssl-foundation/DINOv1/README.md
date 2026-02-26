# DINO v1

Note: This is a fork from https://github.com/facebookresearch/dino


## Train DINO on ImageNet

- sudo docker build -t dino .

- sudo docker run -it --rm --runtime nvidia --shm-size=1g -v /home/nvidia/fssl-foundation/DINOv1:/dino -v /home/nvidia/tiny-imagenet-200/train:/dino/train dino 

- cd src

- python3 dino_wrapper.py


## Visualize DINO attention on ZOD

- python3 visualization_wrapper.py