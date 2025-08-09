
# Setup
```js/java/c#/text
conda create -n FreeGAD python=3.8
conda activate FreeGAD
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install torch-geometric==2.3.1

pip uninstall networkx -y
pip install networkx==2.8.8
pip uninstall dgl -y
pip install dgl==0.9.0
```

# Cite
If you compare with, build on, or use aspects of this work, please cite the following:

```js/java/c#/text
@inproceedings{li2024noise,
author = {Li, Shiyuan and Liu, Yixin and Chen, Qingfeng and Webb, Geoffrey I. and Pan, Shirui},
title = {Noise-Resilient Unsupervised Graph Representation Learning via Multi-Hop Feature Quality Estimation},
year = {2024},
doi = {10.1145/3627673.3679758},
booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
pages = {1255–1265},
}
```
