# 월간 데이콘 예술 작품 화가 분류 AI 경진대회

**Public** : 0.88676, 2/215  
**Private** : 0.88748, 2/215

**Team** : booduck  
**Member** : [김도윤(justist7)](https://github.com/justist7), [김형석(KimHS0915)](https://github.com/KimHS0915), [박근태(sonkt98)](https://github.com/sonkt98), [양윤석(flashult)](https://github.com/flashult), [정선규(SSunQ)](https://github.com/SSunQ)

## Prerequisites

- python 3.8.5
- pytorch 1.12.0
- albumentations 1.3.0
- timm 0.6.11
- scikit-learn 1.1.3

## Description

- 대회 기간 : 2022년 10월 4일 ~ 2022년 11월 14일
- 목표 : 일부분만 주어지는 예술 작품을 화가 별로 분류하는 AI 모델 개발
- Data
  - Class : 화가 50명
  - Train : 화가 50명에 대한 예술 작품 5911개
  - Test : 화가 50명에 대한 예술 작품의 일부분 (약 1/4) 12670개
- Evaluation Metric : Macro F1 score<br><br>
- CV Strategy : stratified 7-fold cross validation
- Backbone : TinyViT
- Loss : CrossEntropy
- Augmentation : Resize, RandomCrop, HorizontalFlip, Cutout, CutMix
- Optimizer : Adam
- Scheduler : CosineAnnealingWarmUpRestarts
- Test Time Augmentation : HorizontalFlip
- Ensemble : 3개의 모델을 Soft voting
  - TinyVit + Cutmix + Stratified 7-Fold OOF
  - TinyVit + Cutout + Stratified 7-Fold OOF
  - TinyVit + Cutout + Validation 없이 데이터를 전부 학습에 사용

## Install

```
pip install -r requirements.txt
```

## How To Use

script 파일을 통해 모델들을 학습시켜 ensemble 결과 추출

```
bash run.sh
```

또는 notebooks의 skf_tta_cutmix, skf_tta_cutout, novalid_ep60_cutout 3가지 ipynb 파일을 실행시켜, 3가지 모델에 대한 test inference logit csv를 추출한 후 ensemble_soft.ipynb를 실행시켜 최종 ensemble output을 추출

## Reference

- [TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/abs/2207.10666)
- [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899)
