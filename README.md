# FreeGAD
 This repository is the official implementation of "FreeGAD: A Training-Free yet Effective Approach for Graph Anomaly Detection"

# Dataset
Some of the larger datasets have been uploaded to Google Drive and can be downloaded via the following [link](https://drive.google.com/drive/folders/1XrXCnSzWrT0h0foQ1FxjBJOyAh5YfCEV?usp=sharing)


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
