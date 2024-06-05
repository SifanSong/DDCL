# Distortion-Disentangled Contrastive Learning (DDCL)

[![Static Badge](https://img.shields.io/badge/WACV-2024-blue)](https://wacv2024.thecvf.com/)
[![Static Badge](https://img.shields.io/badge/DDCL-WACV2024-b31b1b)](https://openaccess.thecvf.com/content/WACV2024/html/Wang_Distortion-Disentangled_Contrastive_Learning_WACV_2024_paper.html)
[![Static Badge](https://img.shields.io/badge/DDCL-PDF-pink)](https://openaccess.thecvf.com/content/WACV2024/papers/Wang_Distortion-Disentangled_Contrastive_Learning_WACV_2024_paper.pdf)
[![Static Badge](https://img.shields.io/badge/Python-3.8.13-blue)]()
[![Static Badge](https://img.shields.io/badge/PyTorch-1.13.0-orange)]()
[![Static Badge](https://img.shields.io/badge/cudatoolkit-11.3.1-1f5e96)]()

Pytorch implementation of **[DDCL (Distortion-Disentangled Contrastive Learning)](https://openaccess.thecvf.com/content/WACV2024/papers/Wang_Distortion-Disentangled_Contrastive_Learning_WACV_2024_paper.pdf)**.

If you find our DDCL useful in your research, please star this repository and consider citing:

```
@inproceedings{wang2024distortion,
  title={Distortion-Disentangled Contrastive Learning},
  author={Wang, Jinfeng and Song, Sifan and Su, Jionglong and Zhou, S Kevin},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={75--85},
  year={2024}
}
```

## Updates

- 30/May/2024: We release a new study which is able to make DDCL much more stable! We rethink the role of invariant and equivariant representations, and extend our idea to improve the efficacy of contrastive learning based on DINO framework. In this case, we have added support for various **mainstream backbone models (ResNet, ViT and VMamba)**. For more detailed information, please refer to **[CLeVER (Contrastive Learning Via Equivariant Representation)](https://github.com/SifanSong/CLeVER)**
- 17/Dec/2023: The code repository is publicly available.
- 29/Nov/2023: Our paper "Distortion-Disentangled Contrastive Learning (DDCL)" was accepted to **WACV2024 (847/2042)** as an **ORAL (53/2042)** paper!

## Abstract

Recently, Positive-pair-Only Contrastive Learning (POCL) has achieved reliable performance without the need to construct positive-negative training sets. The POCL method typically uses a single objective function to extract the distortion invariant representation (DIR) which describes the proximity of positive-pair representations affected by different distortions. This objective function implicitly enables the model to filter out or ignore the distortion variant representation (DVR) affected by different distortions. However, some recent studies have shown that proper use of DVR in contrastive can optimize the performance of models in some downstream domain-specific tasks. In addition, these POCL methods have been observed to be sensitive to augmentation strategies. To address these limitations, we propose a novel POCL framework named Distortion-Disentangled Contrastive Learning (DDCL) and a Distortion-Disentangled Loss (DDL). Our approach is the first to explicitly and adaptively disentangle and exploit the DVR inside the model and feature stream to improve the representation utilization efficiency, robustness and representation ability. Experiments demonstrate our framework’s superiority to Barlow Twins and Simsiam in terms of convergence, representation quality (including transferability and generalization), and robustness on several datasets.

<p align="center">
  <img src="Figures/DDCL_1.png" alt="DDCL1" width="400" />
</p>
<p align="center">
  <img src="Figures/DDCL_2.png" alt="DDCL2" />
</p>


## Getting Started

### Installation

If you intend to run simsiam / DDCL / CLeVER_DDCL with ResNet50, you can simply install the environment using

```bash
conda env create -f DDCL.yml
conda install DDCL
```

### Self-supervised Pre-training

Please follow the configurations and pre-training code provided in `run_train.sh` to conduct experiments on SimSiam, DDCL, and CLeVER_DDCL (*i.e.,* a stable version of DDCL with a novelly proposed regularization loss (*L<sub>PReg</sub>*) to prevent collapse in the new study, [CLeVER](https://github.com/SifanSong/CLeVER)).

### Linear Classification

Please follow the configurations and linear classification codes in `run_train.sh`.

### Performance Evaluation with perturbed input images

Please follow the configurations and evaluation codes in `run_eval.sh`.

### Downstream Classification Task

Please follow the configurations and codes in `run_downstream.sh`.

## Experiments

For Linear Evaluation (ResNet50) of Simsiam, DDCL, DDCL *w/ L<sub>PReg</sub>* on ImageNet-100 with 500 epochs (trained with BAug/CAug/CAug+, and evaluated using Orignal images / ColorJitter / ColorJitter+RandomFlip / ColorJitter+RandomRotation / ColorJitter+RandomRotation+Elastic Transformation), the details are as follows.

| Methods                                                      | Orig.    | CJ       | CJ+Flip  | CJ+Ro    | CJ+Ro+ET |
| ------------------------------------------------------------ | -------- | -------- | -------- | -------- | -------- |
| **Trained by BAug**                                          |          |          |          |          |          |
| Simsiam                                                      | 81.9     | 81.3     | 81.4     | 50.3     | 27.3     |
| DDCL (this paper)                                            | 82.2     | 81.6     | **81.6** | 50.0     | 26.8     |
| DDCL *w/ L<sub>PReg</sub>* ([CLeVER](https://github.com/SifanSong/CLeVER)) | **82.3** | **81.8** | **81.6** | 51.6     | 27.3     |
| **Trained by CAug**                                          |          |          |          |          |          |
| Simsiam                                                      | 79.7     | 79.0     | 79.0     | 77.0     | 51.9     |
| DDCL (this paper)                                            | 80.0     | 79.3     | 79.4     | 77.2     | 48.5     |
| DDCL *w/ L<sub>PReg</sub>* ([CLeVER](https://github.com/SifanSong/CLeVER)) | **80.7** | **80.2** | **80.0** | **77.6** | 48.1     |
| **Trained by CAug+**                                         |          |          |          |          |          |
| Simsiam                                                      | 78.6     | 77.7     | 77.7     | 75.1     | 74.1     |
| DDCL (this paper)                                            | 78.8     | 78.2     | 78.2     | 75.4     | 74.2     |
| DDCL *w/ L<sub>PReg</sub>* ([CLeVER](https://github.com/SifanSong/CLeVER)) | **79.8** | **79.0** | **79.3** | **77.0** | **75.5** |

(\* Compared to default augmentation setting used in Simsiam (*i.e.*, BAug), the CAug has an additional “transforms.RandomRotation(degrees=(-90, 90))” for all input images, and the CAug+ has additional “transforms.RandomRotation(degrees=(-90, 90))” and “transforms.RandomApply([transforms.ElasticTransform(alpha=100.0)], p=0.5)” for all input images.)

## Visualization

Please follow the configurations and codes in `run_vis.sh`.

<img src="Figures/DDCL_5.png" alt="DDCL" style="zoom: 80%;" />

# Acknowledgments

- Thanks Simsiam ([official](https://github.com/facebookresearch/simsiam), [small datasets](https://github.com/Reza-Safdari/SimSiam-91.9-top1-acc-on-CIFAR10)) and Barlow Twins ([official](https://github.com/facebookresearch/barlowtwins?tab=readme-ov-file), [small datasets](https://github.com/IgorSusmelj/barlowtwins)) for their public code and released models. 
