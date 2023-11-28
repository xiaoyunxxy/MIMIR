# MIMIR: Masked Image Modeling for Mutual Information Based Adversarial Robustness

## Environment settings
check requirements.txt

## Install AutoAttack
pip install git+https://github.com/fra31/auto-attack

## Evaluation and Training Commands
check ./script

## Checkpoints
### CIFAR-10

|  Model | Natural | AA | CheckPoint |
|  ----  | ----  | ----  | ----  |
|  ViT-T | 84.82 | 52.96 | [Link](https://drive.google.com/drive/folders/1i40L0tK4UY16DVXljZV2X2fo2PniwsBV?usp=drive_link) |
|  ViT-S | 88.11 | 53.18 | [Link](https://drive.google.com/drive/folders/1C-5I-Gmt3AQA6dIcQ_285LNUHO0m6sZ-?usp=drive_link) |
|  ViT-B | 89.30 | 54.55 | [Link](https://drive.google.com/drive/folders/1yjki5ICIH-vNsSx8RGkKinUhWE6SNp1m?usp=drive_link) |
|  ConViT-T | 80.74 | 45.04 | [Link](https://drive.google.com/drive/folders/14gHxaT_fn94quZagNv-TR8WTuEvRh39D?usp=drive_link) |
|  ConViT-S | 87.49 | 52.54 | [Link](https://drive.google.com/drive/folders/1YEluyokNSP1kO_Yxs-cPI5HMe4UJJsJq?usp=drive_link) |
|  ConViT-B | 89.30 | 55.64 | [Link](https://drive.google.com/drive/folders/1Loyoy8GvS1mxmK7QrKLkyci0QrSkakQ4?usp=drive_link) |


### ImageNet-1K
### eps 2
|  Model | Natural | PGD 20 | CheckPoint |
|  ----  | ----  | ----  | ----  |
|  ViT-S | 74.60 | 54.56 | [Link](https://drive.google.com/drive/folders/1wSG3J1JwZccMhiAigMpD9KoHzEKxY9xr?usp=drive_link) |
|  ViT-B | 75.88 | 55.42 | [Link](https://drive.google.com/drive/folders/1BgDoMPnq7M5Y34mgHdBX1WoNsZGLJR1W?usp=drive_link) |

#### eps 4
|  Model | Natural | PGD 20 | CheckPoint |
|  ----  | ----  | ----  | ----  |
|  ViT-S | 71.29 | 40.98 | [Link](https://drive.google.com/drive/folders/1c42Y_1pdC5iRTyv66P6tEehetuh-r1v9?usp=drive_link) |
|  ViT-B | 73.22 | 41.26 | [Link](https://drive.google.com/drive/folders/1YMP5Mk3mBcdg8y2A1e0nmu3XHOtdKwHg?usp=drive_link) |


### MIMIR Pre-train checkpoints
|  Dataset | Model | CheckPoint |
|  ----  | ----  | ---- |
| CIFAR-10 |  ViT-T | [Link](https://drive.google.com/drive/folders/1ogRUP_vKRnG9XvTwB0xLqVBf3Ag3Sk-f?usp=drive_link)  |
| CIFAR-10 |  ViT-S | [Link](https://drive.google.com/drive/folders/1DLWuUH1egDU3axXz9Gx2yEFmnE0JBzBX?usp=drive_link)  |
| CIFAR-10 |  ViT-B | [Link](https://drive.google.com/drive/folders/1WI1b6N_tP23INFvrAOYdTc8bd6aROsg8?usp=drive_link)  |
| CIFAR-10 |  ConViT-T | [Link](https://drive.google.com/drive/folders/1YvQz2QUcc1Z9weg9FHk-fTQG9qwSEhWR?usp=drive_link)  |
| CIFAR-10 |  ConViT-S | [Link](https://drive.google.com/drive/folders/1LGz5YoBnnm32z3y6pT_dUM9cHzQBLL-2?usp=drive_link)  |
| CIFAR-10 |  ConViT-B | [Link](https://drive.google.com/drive/folders/1mJekoZw2imovMP7fhGoApfQ6uOlGvYWT?usp=drive_link)  |
| ImageNet-1K |  ViT-S | [Link](https://drive.google.com/drive/folders/1eXPQxNwJXyBknb42sq1yT7hCj31SjqTS?usp=drive_link)  |
| ImageNet-1K |  ViT-B | [Link](https://drive.google.com/drive/folders/1nUTPSelq18h3k7xe9CBv6yx8CHHYPotM?usp=drive_link)  |


## Acknowlegements
This repository is built upon the following repositories:
https://github.com/facebookresearch/mae
https://github.com/wzekai99/DM-Improves-AT
https://github.com/yuxi120407/DIB
https://github.com/choasma/HSIC-bottleneck
