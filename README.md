# Klein Model for Hyperbolic Neural Networks

This repository provides the official implementation of hyperbolic neural networks (HNNs) based on the Klein model from the following paper.


> Yidan Mao, Jing Gu, Marcus C. Werner, and Dongmian Zou. Klein Model for Hyperbolic Neural Networks. arXiv preprint arXiv:2410.16813. 


## 1. Environment
* numpy 1.21.6
* scikit-learn 0.20.3
* torch 1.8.1
* torchvision 0.9.1
* networkx 2.2
For more specific information, please see `environment.yml`.


## 2. Usage
Before training, run 

`source set_env.sh`

to create environment variables that are used in the code.


## 3. Examples

We provide examples of scripts to train HNNs in the Klein model. In the examples below, we used a fixed random seed set to 1234 for reproducibility. Results may vary slightly based on the machine used. To reproduce results in the paper, run each script for 10 random seeds and average the results.

* For Texas dataset

`CUDA_VISIBLE_DEVICES=1 nohup python train.py --task nc --dataset texas --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold Klein --log-freq 1 > log.file 2>&1 >&1 &`

* For Wisconsin dataset

`CUDA_VISIBLE_DEVICES=1 nohup python train.py --task nc --dataset wisconsin --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0.0001 --manifold Klein --log-freq 1 > log.file 2>&1 >&1 &`

* For Chameleon dataset

`CUDA_VISIBLE_DEVICES=1 nohup python train.py --task nc --dataset chameleon --model HNN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0.001 --manifold Klein --log-freq 1 > log.file 2>&1 >&1 &`

* For Actor dataset

`CUDA_VISIBLE_DEVICES=1 nohup python train.py --task nc --dataset film --model HNN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Klein --log-freq 1 > log.file 2>&1 >&1 &

* For Cora dataset

`CUDA_VISIBLE_DEVICES=1 nohup python train.py --task nc --dataset cora --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.2 --weight-decay 0.001 --manifold Klein --log-freq 1 > log.file 2>&1 >&1 &`

* For Pubmed dataset

`CUDA_VISIBLE_DEVICES=1 nohup python train.py --task nc --dataset pubmed --model HNN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.2 --weight-decay 0.001 --manifold Klein --log-freq 1 > log.file 2>&1 >&1 &`


## 4. File Descriptions
`data/`: Datasets

`layers/`: Hyperbolic layers

`manifolds/`: Manifold calculations
* `manifolds/klein.py` contains the key operations in the Klein model.

`models/`: Hyperbolic models

`optimizers/`: Optimization on manifolds

`utils/`: Utility files

`train.py`: Training script

`config.py`: Hyperparameter settings


## Citation
```
Yidan Mao, Jing Gu, Marcus C. Werner, and Dongmian Zou. Klein Model for Hyperbolic Neural Networks. In _NeurIPS 2024 Workshop on Symmetry and Geometry in Neural Representation_s (NeurReps), 2024.
```

or 

```
@article{mao2024klein,
  title={Klein Model for Hyperbolic Neural Networks},
  author={Mao, Yidan and Gu, Jing and Werner, Marcus C and Zou, Dongmian},
  journal={arXiv preprint arXiv:2410.16813},
  year={2024}
}
```

## Reference
For the construction of hyperbolic models, we utilized the code available at https://github.com/HazyResearch/hgcn.
