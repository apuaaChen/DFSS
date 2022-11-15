# Dynamic N:M Fine-grained Structured Sparse Attention Mechanism

This repo contains the artifact for our PPoPP paper [Dynamic N:M Fine-grained Structured Sparse Attention Mechanism](https://arxiv.org/pdf/2203.00091.pdf). 

## Using Docker
We use NGC pytorch container 21.06. To build the container, run
```shell
cd docker && bash build.sh
```
To launch the container, run
```shell
cd .. && bash docker/launch.sh
```
The code is mounted to `/workspace/dfss`.

## Installation
