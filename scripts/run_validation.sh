#!/bin/bash

pth="/home/nathan/histo-seg/semantic-pca"

dataset="xval_set_0_512"
python validate.py $pth/data/$dataset/val/mask \
$pth/analysis_crf/$dataset \
$pth/analysis_crf/report.$dataset.txt

dataset="xval_set_1_512"
python validate.py $pth/data/$dataset/val/mask \
$pth/analysis_crf/$dataset \
$pth/analysis_crf/report.$dataset.txt

dataset="xval_set_2_512"
python validate.py $pth/data/$dataset/val/mask \
$pth/analysis_crf/$dataset \
$pth/analysis_crf/report.$dataset.txt

dataset="xval_set_3_512"
python validate.py $pth/data/$dataset/val/mask \
$pth/analysis_crf/$dataset \
$pth/analysis_crf/report.$dataset.txt

dataset="xval_set_4_512"
python validate.py $pth/data/$dataset/val/mask \
$pth/analysis_crf/$dataset \
$pth/analysis_crf/report.$dataset.txt

dataset="xval_set_0_1024"
python validate.py $pth/data/$dataset/val/mask \
$pth/analysis_crf/$dataset \
$pth/analysis_crf/report.$dataset.txt

dataset="xval_set_1_1024"
python validate.py $pth/data/$dataset/val/mask \
$pth/analysis_crf/$dataset \
$pth/analysis_crf/report.$dataset.txt

dataset="xval_set_2_1024"
python validate.py $pth/data/$dataset/val/mask \
$pth/analysis_crf/$dataset \
$pth/analysis_crf/report.$dataset.txt

dataset="xval_set_3_1024"
python validate.py $pth/data/$dataset/val/mask \
$pth/analysis_crf/$dataset \
$pth/analysis_crf/report.$dataset.txt

dataset="xval_set_4_1024"
python validate.py $pth/data/$dataset/val/mask \
$pth/analysis_crf/$dataset \
$pth/analysis_crf/report.$dataset.txt
