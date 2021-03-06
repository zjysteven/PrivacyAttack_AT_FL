# Exploiting Adversarial Training models to compromise FL's privacy

This repository contains the code necessary to replicate the results of our paper:

***Privacy Leakage of Adversarial Training Models in Federated Learning Systems***

[*CVPR'22 Workshop: The Art of Robustness*](https://artofrobust.github.io/)

[Jingyang Zhang](https://zjysteven.github.io/), Yiran Chen, Hai Li @ [Duke CEI Lab](https://cei.pratt.duke.edu/)

Check out the paper [here](https://openaccess.thecvf.com/content/CVPR2022W/ArtOfRobust/html/Zhang_Privacy_Leakage_of_Adversarial_Training_Models_in_Federated_Learning_Systems_CVPRW_2022_paper.html)!!!


## Overview
<p align="center">
<img src='/figures/overview_new.png' width='550'>
</p>

Adversarial Training (AT) is crucial for obtaining deep neural networks that are robust to adversarial attacks, yet
recent works found that it could also make models more vulnerable to privacy attacks. In this work, we further reveal
this unsettling property of AT by designing a novel privacy attack that is practically applicable to the privacy-sensitive
Federated Learning (FL) systems. **Using our method, the attacker can exploit AT models in the FL system to accurately reconstruct users’ private training images even when the training batch size is large,** despite that previously large batch training was thought to be able to protect the privacy. In other words, **we demonstrate that any FL clients that perform AT are putting their privacy at risk**. The above figure is an overview of our attack.
See our paper for details!

<p align="center">
<img src='/figures/demo.png' width='650'>
</p>
This figure visualizes the reconstructed images using Adversarial Training models (first row in each pair) and the ground-truth ones. The attacker can pretty accurately recover the training images of FL clients and thus compromise their privacy.


## Get started
### Environment
Follow the commands below to set up the environment.

1. Clone the repo: `git clone https://github.com/zjysteven/PrivayAttack_AT_FL.git`

2. Create a conda environment
```
conda create -n AT-privacy python=3.8
conda activate AT-privacy
python -m pip install -r requirements.txt
```

### Dataset
We use [ImageNet](https://www.image-net.org/) as the dataset. To run our experiments, make sure that you download the ImageNet and have `train/`, `val/` subfolder inside the root directory. 

### Models
We use pre-trained models from [robust-transfer](https://github.com/microsoft/robust-models-transfer) repo. `download_pretrained_models.sh` is a sample download script. 

## Reproducing experiments
All experiments can be reproduced by running `main.py`. We provide a sample script `main.sh`. Remember to change the directory of ImageNet to your own version.

## Reference
If you find our work/code helpful, please consider citing our work.
```
@InProceedings{Zhang_2022_CVPR,
    author    = {Zhang, Jingyang and Chen, Yiran and Li, Hai},
    title     = {Privacy Leakage of Adversarial Training Models in Federated Learning Systems},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {108-114}
}
```
