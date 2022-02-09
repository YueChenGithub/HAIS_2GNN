# HAIS_2GNN: 3D Visual Grounding with Graph and Attention


This repository is for the HAIS_2GNN research project.

Tao Gu, Yue Chen

![](docs/Poster.png)


## Introduction

The motivation of this project is to improve the accuracy of 3D visual grounding. In this report, we propose a new model, named HAIS 2GNN based on the InstanceRefer model, to tackle the problem of insufficient connections between instance proposals. Our model incorporates a powerful instance segmentation model HAIS and strengthens the instance features by the structure of graph and attention, so that the text and point cloud can be better matched together. Experiments confirm that our method outperforms the InstanceRefer on ScanRefer validation datasets.


## Setup
The code is tested on Ubuntu 20.04.3 LTS with **Python 3.9.7 PyTorch 1.10.1 CUDA 11.3.1** installed. 

```shell
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch
```

Install the necessary packages listed out in `requirements.txt`:
```shell
pip install -r requirements.txt
```
After all packages are properly installed, please run the following commands to compile the [**torchsaprse v1.4.0**](https://github.com/mit-han-lab/torchsparse):
```shell
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```
__Before moving on to the next step, please don't forget to set the project root path to the `CONF.PATH.BASE` in `lib/config.py`.__


### Data preparation
1. Download the ScanRefer dataset and unzip it under `data/`. 
2. Downloadand the preprocessed [GLoVE embeddings (~990MB)](http://kaldir.vc.in.tum.de/glove.p) and put them under `data/`.
3. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `data/scannet/scans/` (Please follow the [ScanNet Instructions](data/scannet/README.md) for downloading the ScanNet dataset). After this step, there should be folders containing the ScanNet scene data under the `data/scannet/scans/` with names like `scene0000_00`
4. Used official and pre-trained [HAIS](https://github.com/hustvl/HAIS) generate panoptic segmentation in `PointGroupInst/`. We will provide the pre-trained data soon.
5. Pre-processed instance labels, and new data should be generated in  `data/scannet/pointgroup_data/`
```shell
cd data/scannet/
python prepare_data.py --split train --pointgroupinst_path [YOUR_PATH]
python prepare_data.py --split val   --pointgroupinst_path [YOUR_PATH]
python prepare_data.py --split test  --pointgroupinst_path [YOUR_PATH]
```
Finally, the dataset folder should be organized as follows.
```angular2
InstanceRefer
├── data
│   ├── glove.p
│   ├── ScanRefer_filtered.json
│   ├── ...
│   ├── scannet
│   │  ├── meta_data
│   │  ├── pointgroup_data
│   │  │  ├── scene0000_00_aligned_bbox.npy
│   │  │  ├── scene0000_00_aligned_vert.npy
│   │  ├──├──  ... ...
```

### Training
Train the InstanceRefer model. You can change hyper-parameters in `config/InstanceRefer.yaml`:
```shell
python scripts/train.py --log_dir HAIS_2GNN
```

### Evaluation
You need specific the `use_checkpoint` with the folder that contains `model.pth` in `config/InstanceRefer.yaml` and run with:
```shell
python scripts/eval.py
```

### Pre-trained Models
| Input | ACC@0.25 | ACC@0.5 | Checkpoints
|--|--|--|--|
| xyz+rgb |  39.24  | 33.66 |  will be released soon

## TODO

- [ ] Add pre-trained HAIS dataset.
- [ ] Release pre-trained model.
- [ ] Merge HAIS in an end-to-end manner.
- [ ] Upload to ScanRefer benchmark

## Changelog
02/09/2022: Released the HAIS_2GNN

## Acknowledgement
This work is a research project conducted by Tao Gu and Yue CHen for ADL4CV:Visual Computing course at the Technical University of Munich.

We acknowledge that our work is based on [ScanRefer](https://github.com/daveredrum/ScanRefer), [InstanceRefer](https://github.com/CurryYuan/InstanceRefer), [HAIS](https://github.com/hustvl/HAIS),[torchsaprse](https://github.com/mit-han-lab/torchsparse), and [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric).


