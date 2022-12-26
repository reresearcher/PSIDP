# PSIDP:Unsupervised Deep Hashing with Pretrained Semantic Information Distillation and Preservation

# --Neurocomputing@2022

## Environment:

Archlinux 5.14.12-arch1-1

Python 3.9.7

torch 1.10.1

torchvision 0.11.2

pillow 8.3.1

numpy 1.20.3

scipy 1.7.1

scikit-learn 0.24.2

argparse 1.1

tqdm 4.62.2

loguru 0.5.3


## How to use:

1. Download the Flickr-25k dataset:

    Baidu Cloud: https://pan.baidu.com/s/1gv7PkawQitYoUogndUEPXg 
    password: a91q 

2. Put the flickr25k.zip in the ./Datasets and unzip it.

3. ```python
   python run.py --train --dataset 'flickr25k' --code-length 16 --num-query 2000 --num-train 5000
   ```

## About Flickr-1M dataset

The 400-dim labels of large-scale Flickr-1M (1,000,000 images) dataset are also provided:

  Baidu Cloud: https://pan.baidu.com/s/1-2AJG_3tPv5WKig7GwtkmQ 
  password: 258w 

## If you find PSIDP useful, please cite:
```
@article{shi2022psidp,
  title={PSIDP: Unsupervised deep hashing with pretrained semantic information distillation and preservation},
  author={Shi, Yufeng and You, Xinge and Zhao, Yue and Xu, Jiamiao and Ou, Weihua and Zheng, Feng and Peng, Qinmu},
  journal={Neurocomputing},
  volume={502},
  pages={1--14},
  year={2022},
  publisher={Elsevier}
}
```

## If you have any questions, please feel free to contact me yufengshi17@hust.edu.cn
