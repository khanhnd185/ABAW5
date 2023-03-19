
# ABAW5 Repo

## Introduction

5th Workshop and Competition on Affective Behavior Analysis in-the-wild ([ABAW5](https://ibug.doc.ic.ac.uk/resources/cvpr-2023-5th-abaw/))

Our report for the competition is [here](https://arxiv.org/abs/2303.09293). The feature and trained models can be found in this [archive](https://ejnu-my.sharepoint.com/:f:/g/personal/nguyendangkhanh_jnu_ac_kr/EkrSGxjTG4JGvv49tw5vFngB_Kf5ia4TnsgblfJmLbt-9A?e=bb1nI8)


# Dependency

We borrow EfficientNet of Savchenko, we must use the exact version of timm:
```
pip install timm==0.4.5
```

The pre-trained EfficientNet B0 on  Facial Behavior Tasks of Savchenko is in [this project](https://github.com/HSE-asavchenko/face-emotion-recognition)

# Model Architecture

![image info](./arc.png)

## Result

Evaluation metrics of Expression classification on Aff-Wild2 Validation set
| Model | F1 |
| --- | --- |
| Effnet+MLP | 0.3327 |
| Effnet+Transformer Encoder (N=4, h=4) | 0.3615 |
| Effnet+Transformer Encoder (N=4, h=4), Augment (1) | 0.4400 |
| Effnet+Transformer Encoder (N=4, h=8) , Augment (2) | 0.4424 |
| Effnet+Transformer Encoder (N=6, h=4) , Augment (3) | 0.4555 |
| Average Ensemble (1)(2) | 0.4663 |
| Average Ensemble (1)(3) | 0.4672 |
| Average Ensemble (3)(2) | 0.4729 |
| Average Ensemble (1)(2)(3) | **0.4775** |


Evaluation metrics of Valence-Arousal estimation on Aff-Wild2 Validation set
| Model | F1 |
| --- | --- |
| Effnet+Transformer Encoder (N=4, h=4) (1) | 0.48296 |
| Effnet+Transformer Encoder (N=4, h=8) (2) | 0.48819 |
| Effnet+Transformer Encoder (N=6, h=4) (3) | 0.47389 |
| Average Ensemble (1)(2) | 0.49684 |
| Average Ensemble (1)(3) | 0.49679 |
| Average Ensemble (3)(2) | 0.49874 |
| Average Ensemble (1)(2)(3) | **0.50290** |

Evaluation metrics of Action Unit Detection on Aff-Wild2 Validation set
| Model | F1 |
| --- | --- |
| Effnet+Transformer Encoder (N=4, h=4) (1) | 0.51696 |
| Effnet+Transformer Encoder (N=4, h=8) (2) | 0.51146 |
| Effnet+Transformer Encoder (N=6, h=4) (3) | 0.51192 |
| Average Ensemble (1)(2) | 0.51960 |
| Average Ensemble (1)(3) | 0.52021 |
| Average Ensemble (3)(2) | 0.51709 |
| Average Ensemble (1)(2)(3) | **0.52085** |


