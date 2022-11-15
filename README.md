# Dynamic N:M Fine-grained Structured Sparse Attention Mechanism

This repo contains the artifact for our PPoPP paper [Dynamic N:M Fine-grained Structured Sparse Attention Mechanism](https://arxiv.org/pdf/2203.00091.pdf). 

## Get Source Code
Get source code with
```shell
https://github.com/apuaaChen/DFSS.git
```
Get the submodules
```shell
git submodule update --init --recursive
```

## Using Docker
We use NGC pytorch container 21.06. To build the container, run
```bash
cd docker && bash build.sh
```
To launch the container, run
```bash
cd .. && bash docker/launch.sh
```
The code is mounted to `/workspace/dfss`.

## Installation
Our package `pydfss` can be installed with
```bash
cd /workspace/dfss && bash install.sh
```

## Speedup under different sequence length
We provide the script to reproduce the attention speedup under different sequence length with `bfloat16` data type. 
```bash
python benchmark.py
```
As mentioned in the paper, we only compare the `QK^T`, `Softmax` and `AV` in this script, as the optimizations in other parts are orthogonal to our DFSS. The expected result could be around
```
attention speedup: 1.38 ~ 1.86
```

## Accuracy 
We provide training and inference scripts of BERT-large on SQuAD v1.1 with DFSS 2:4 under `bfloat16` data type (Table 2 in the paper). The script requires 2 A100 GPUs, and it takes about 1.5 hour to finish.
```bash
mkdir ckpt && python bert_squad_finetuning.py
``` 
Expected result would be
```
F1 score on BERT-large SQuAD v1.1
Transformer: 93.10, DFSS 2:4: 93.19
```