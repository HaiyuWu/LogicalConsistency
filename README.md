# Logical-Consistency-and-Greater-Descriptive-Power-for-Facial-Hair-Attribute-Learning
## Paper accepted to the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2023

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
<figure>
  <img src="./teaser.png" style="width:100%">
  <figcaption>Figure 1: We provide richer descriptions on facial hair that covers Beard Area, Beard Length, Mustache, Sideburns, and Bald. We first consider the logical consistency of the predictions for multi-label classification task.</figcaption>
</figure>

## Import Performance Update
### Accuracy

| Methods         | $Acc_{avg}$ | $Acc^{n}_{avg}$ | $Acc^{p}_{avg}$ |
| :-------------- | :---------: | :-------------: | :-------------: |
| BCE$^*$         |    91.16    |      94.71      |      68.00      |
| BCE-MOON$^*$    |    89.24    |      91.01      |      80.96      |
| BF$^*$          |    90.19    |      97.38      |      51.2       |
| BCE+LCP$^*$     |             |                 |                 |
|                 |             |                 |                 |
| BCE$^*$         |    57.79    |      58.95      |      47.05      |
| BCE-MOON$^*$    |    49.58    |      50.53      |      34.84      |
| BF$^*$          |    23.38    |      23.91      |      18.63      |
| BCE+LCP$^*$     |             |                 |                 |
|                 |             |                 |                 |
| BCE+LC$^*$      |    85.98    |      88.30      |      67.76      |
| BCE-MOON+LC$^*$ |    52.32    |      53.40      |      36.46      |
| BF+LC$^*$       |    90.48    |      93.19      |      67.33      |
| BCE+LCP+LC$^*$  |             |                 |                 |



### Logical Consistency



## TL;DR
This repository provides a facial hair dataset, FH37K, which describes the facial hair in Area, Length, Connectedness dimensions. In addition, a method that can force the model to make predictions logical. See details below.

## Table of contents

<!--ts-->
- [Paper details](#paper-details)
  * [Abstract](#abstract)
  * [Citation](#citation)
  * [Credit](#credit)
  * [Attribute Definition](#attribute-definition)
  * [Bias Aware test set](#bias-aware-test-set)
- [Installation](#installation)
- [Training](#training)
  * [Prepare FH37K dataset](#prepare-fh37k-dataset)
  * [Train](#train)
- [Testing](#testing)
  * [Accuracy](#Accuracy)
  * [Logical Consistency](#logical-consistency)
- [License](#license)
  <!--te-->

## Paper details
[Haiyu Wu](https://haiyuwu.netlify.app/), Grace Bezold, Aman Bhatta, [Kevin W. Bowyer](https://www3.nd.edu/~kwb/), "*Logical Consistency and Greater Descriptive Power for Facial Hair Attribute Learning*", CVPR, 2023, [arXiv:2302.11102](https://arxiv.org/abs/2302.11102)

### Abstract
> Face attribute research has so far used only simple binary attributes for facial hair; e.g., beard / no beard. We have created a new, more descriptive facial hair annotation scheme and applied it to create a new facial hair attribute dataset, FH37K. Face attribute research also so far has not dealt with logical consistency and completeness. For example, in prior research, an image might be classified as both having no beard and also having a goatee (a type of beard). We show that the test accuracy of previous classification methods on facial hair attribute classification drops significantly if logical consistency of classifications is enforced. We propose a logically consistent prediction loss, LCPLoss, to aid learning of logical consistency across attributes, and also a label compensation training strategy to eliminate the problem of no positive prediction across a set of related attributes. Using an attribute classifier trained on FH37K, we investigate how facial hair affects face recognition accuracy, including variation across demographics. Results show that similarity and difference in facial hairstyle have important effects on the impostor and genuine score distributions in face recognition.

### Attribute Definition
The definition and examples of each attribute can be found in [Google Document](https://docs.google.com/document/d/1My0catzzRc5wzxIo2-ozDaDuZrv3Z_wG/edit#).

### Bias Aware test set
The [BA-test](https://github.com/HaiyuWu/BA-test-dataset) dataset used in the demographic face recognition accuracy disparity analyses part is available now. However, note that this is the cleaner and more controlled version, so that the reproduced results should have slight difference, but it does not change the observations.

### Citation
If you use any part of our code or model, please cite our paper.
```
@inproceedings{wu2023logical,
  title={Logical Consistency and Greater Descriptive Power for Facial Hair Attribute Learning},
  author={Wu, Haiyu and Bezold, Grace and Bhatta, Aman and Bowyer, Kevin W},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8588--8597},
  year={2023}
}
```
### Credit
If you use the FH37K dataset, please **ALSO** cite WebFace260M.
```
@inproceedings {zhu2021webface260m,
  title=  {WebFace260M: A Benchmark Unveiling the Power of Million-scale Deep Face Recognition},
  author=  {Zheng Zhu, Guan Huang, Jiankang Deng, Yun Ye, Junjie Huang, Xinze Chen, Jiagang Zhu, Tian Yang, Jiwen Lu, Dalong Du, Jie Zhou},
  booktitle=  {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year=  {2021}              
}
```

## Installation
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

Install dependecies with Python 3.
```
pip install -r requirements.txt
```

## Training
### Prepare FH37K dataset
Since we have not gotten the permission to distribute the images from CelebA, 
you need to download the ***cropped and aligned*** dataset from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). **The WebFace260M portion is available now!**, you can download it from [Drive](https://drive.google.com/drive/folders/17VKdbEE2iCwxCTdsUucCne762t2h01VV?usp=sharing). 
Since it has been separated to train/val/test, you can use the downloaded folder as the destination to run the following code to collect the rest of FH37k.
```
python fh37k_collection.py \
-celeba /path/to/celeba/folder \
-pf ./FH37K/label_partition.csv \
-lf ./FH37K/facial_hair_annotations.csv
-d path/to/webface/portion/folder
```
After you have the FH37K dataset, using the [image2lmdb.py](./utils/image2lmdb.py) to generate a LMDB dataset (train and val).
```
python utils/image2lmdb.py \
-im ./FH37K/train \
-l ./FH37K/facial_hair_annotations.csv \
-t train \
-d fh37k_lmdb_dataset
```
### Train
Once the LMDB train/val file are ready, simply run the following script to start training
```
python -u train_with_lmdb.py \
-td ./fh37k_lmdb_dataset/train.lmdb \
-vd ./fh37k_lmdb_dataset/val.lmdb \
-sr ./model_weights/ \
-m resnet50 \
-lf ./model_weights/resnet50_loss.npy \
-bs 256 \
-lr 1e-3 \
-e 50 \
-a 24 \
-l 0.5 \
-pt
```
## Testing
To evaluate with the pretrained model, download the model from [Model Zoo](https://drive.google.com/drive/folders/1ttUaN3kOHJ9GYLz0nQd19hDOIm9AHeTO?usp=sharing),
and put models to ./weights. 

| models     | Overall | Negative | Positive |
| ---------- | ------- | -------- | -------- |
| ResNet50   | 89.89   | 92.65    | 70.23    |
| SE-ResNeXt | 89.71   | 92.31    | 71.33    |

This project evaluate the model in two aspects - ***Accuracy*** and ***Logical consistency***.

### Accuracy
To evaluate the accuracy on test set, simply edit the bash script [image_folder_test.sh](image_folder_test.sh) and run
```
bash image_folder_test.sh
```
### Logical Consistency
To evaluate the logical consistency in the real-world, you can download the test set from [Drive](https://drive.google.com/drive/folders/1ttUaN3kOHJ9GYLz0nQd19hDOIm9AHeTO?usp=sharing). 
After you have this test set, run [file_path_extractor.py](./file_path_extractor.py) to collect the image paths.
``` 
python file_path_extractor.py -s /path/to/extracted/folder -d . -end_with jpg
```

Once the image paths have been collected in a .txt file, there are two ways to do the prediction binarization. 

1. Edit the paths in [image_file_test.sh](./image_file_test.sh) and run the bash script to save the predictions
``` 
bash image_file_test.sh
```
2. Edit the paths in [image_file_test.sh](./image_file_test.sh), replace ```-lc``` with ```-rc``` to save the raw confidences.
  This process provides a flexibility to use confidence for image selection.
```
bash image_file_test.sh
```
Then convert the raw confidences to binary predictions and save it in another file
``` 
python Scripts/rc2binary.py -rf /path/to/confidence/file -sf /path/to/destination  -o file_name
```
*Tip: adding ```-c``` to enable label compensation strategy.*

After getting the binarized predictions, calculating the fail rate with the logical consistency checked
``` 
python Scripts/check_impossible_ratio.py -bt /your/prediction/file
```

## License
Check [license](./license.md) for license details.
