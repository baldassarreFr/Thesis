# DINO v1

Note: This is a fork from https://github.com/facebookresearch/dino


## Train & Evaluate Object Detector on ZOD

- sudo docker build -t dino .

- sudo docker run -it --rm --runtime nvidia --shm-size=1g -v /home/nvidia/fssl-foundation/evaluation:/dino -v /mnt/ZODversions/ZOD256:/dino/ZOD dino 

- cd src

- python3 fasterrcnn.py
