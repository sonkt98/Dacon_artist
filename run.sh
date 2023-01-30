#!bin/bash

python train.py --epochs 60 --model TinyVit21m384Cls --scheduler CosineAnnealingWarmUpRestarts --augmentation AugmentationV3 --resize 800 --crop_size 384 --batch_size 32 --no_valid --name model1

python inference.py --model TinyVit21m384Cls --img_size 384 --model_path output/model/model1/latest.pt --mode logit

mkdir -p output/ensemble
mv output/submission/logit.csv output/ensemble/logit1.csv

python train.py --epochs 50 --model TinyVit21m384 --scheduler CosineAnnealingWarmUpRestarts --augmentation AugmentationV2 --resize 800 --crop_size 384 --batch_size 32 --kfold --stratified --n_splits 7 --tta --oof --early_stopping 10

mv output/submission/oof_logit.csv output/ensemble/logit2.csv

python train.py --epochs 50 --model TinyVit21m384 --scheduler CosineAnnealingWarmUpRestarts --augmentation AugmentationV1 --resize 800 --crop_size 384 --batch_size 32 --cutmix --kfold --stratified --n_splits 7 --tta --oof --early_stopping 10

mv output/submission/oof_logit.csv output/ensemble/logit3.csv

python ensemble.py --ensemble_path output/ensemble/ --voting soft
