#!/bin/bash

pth="/home/nathan/histo-seg/semantic-pca"
expt="analysis_segnet_basic"

dataset="xval_set_0"
testdir=$dataset"_512"
python validate.py $pth/data/$dataset/val/mask \
$pth/$expt/$testdir \
$pth/$expt/report.$testdir.txt

testdir=$dataset"_1024"
python validate.py $pth/data/$dataset/val/mask \
$pth/$expt/$testdir \
$pth/$expt/report.$testdir.txt

dataset="xval_set_1"
testdir=$dataset"_512"
python validate.py $pth/data/$dataset/val/mask \
$pth/$expt/$testdir \
$pth/$expt/report.$testdir.txt

testdir=$dataset"_1024"
python validate.py $pth/data/$dataset/val/mask \
$pth/$expt/$testdir \
$pth/$expt/report.$testdir.txt

dataset="xval_set_2"
testdir=$dataset"_512"
python validate.py $pth/data/$dataset/val/mask \
$pth/$expt/$testdir \
$pth/$expt/report.$testdir.txt

testdir=$dataset"_1024"
python validate.py $pth/data/$dataset/val/mask \
$pth/$expt/$testdir \
$pth/$expt/report.$testdir.txt

dataset="xval_set_3"
testdir=$dataset"_512"
python validate.py $pth/data/$dataset/val/mask \
$pth/$expt/$testdir \
$pth/$expt/report.$testdir.txt

testdir=$dataset"_1024"
python validate.py $pth/data/$dataset/val/mask \
$pth/$expt/$testdir \
$pth/$expt/report.$testdir.txt

dataset="xval_set_4"
testdir=$dataset"_512"
python validate.py $pth/data/$dataset/val/mask \
$pth/$expt/$testdir \
$pth/$expt/report.$testdir.txt

testdir=$dataset"_1024"
python validate.py $pth/data/$dataset/val/mask \
$pth/$expt/$testdir \
$pth/$expt/report.$testdir.txt
