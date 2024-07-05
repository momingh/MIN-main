# MIN: Multi-stage Interactive Network for Multimodal Recommendation

<!-- PROJECT LOGO -->

## Introduction

This is the Pytorch implementation for our MIN paper:

>MIN: Multi-stage Interactive Network for
Multimodal Recommendation



## Dataset

We provide three processed datasets: Baby, Sports, Clothing.

Download from Google Drive: [Baby/Sports/Clothing](https://drive.google.com/file/d/12GUVxa6v3L3JbCd0KJTwM1ltd1oA3MHI/view?usp=sharing)

## Training
  ```
  cd ./src
  python main.py
  ```

In **main.py**, modify the dataset, and in **/config/model/MIN.yaml**, modify the hyperparameters. The optimal hyperparameters for each dataset are as follows:
```angular2html
Baby

cl_weight: [0.005]
epsilon: [5]
lambda1: [0.001]
pvn_weight: [0.01]
```

```angular2html
Sports

cl_weight: [0.01]
epsilon: [2]
lambda1: [0.05]
pvn_weight: [0.01]
```

```angular2html
Clothing

cl_weight: [0.01]
epsilon: [0.2]
lambda1: [0.01]
pvn_weight: [0.005]
```

## Acknowledgement
The structure of this code is  based on [MMRec](https://github.com/enoche/MMRec) and [MENTOR](https://github.com/Jinfeng-Xu/MENTOR). Thank for their work.