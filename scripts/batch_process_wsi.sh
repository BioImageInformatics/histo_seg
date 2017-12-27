#/bin/bash

set -e

## New way - using histoseg_batch and tfmodels
settings_file=example/resnet_10x_tf.pkl

# svs_dir=/home/nathan/data/pca_wsi
# svs_dir=/media/nathan/DATA/histo-seg-data/slide_for_testing_SPIE_2017
svs_dir=/media/nathan/DATA/_durham_slides/not_processed

# python core/histoseg_batch.py --source_dir=$svs_dir --settings=$settings_file
#python core/histoseg_batch.py --source_dir=$svs_dir --settings=$settings_file --random=20

## Below is for single slide histoseg.py
for svs in $( ls $svs_dir/*svs ); do
  echo $svs
done

logfile="/home/nathan/histo-seg/durham/resnet10x/log_20171222.txt"
ls $svs_dir/*svs | parallel --jobs 2 \
"python /home/nathan/histo-seg/v2/core/histoseg.py --slide={} --settings=$settings_file" | tee \
$logfile

# logfile="/home/nathan/histo-seg/semantic-pca/analysis_wsi/segnet_basic_crf/log.txt"
# ls /media/nathan/DATA/histo-seg-data/slide_for_testing_SPIE_2017/*svs | parallel --jobs 2 \
# "python /home/nathan/histo-seg/v2/core/histoseg.py --slide={} --settings=/home/nathan/histo-seg/v2/example/segnet_basic_crf_settings.pkl" | tee \
# $logfile
#
# logfile="/home/nathan/histo-seg/semantic-pca/analysis_wsi/fcn8s/log.txt"
# ls /media/nathan/DATA/histo-seg-data/slide_for_testing_SPIE_2017/*svs | parallel --jobs 2 \
# "python /home/nathan/histo-seg/v2/core/histoseg.py --slide={} --settings=/home/nathan/histo-seg/v2/example/fcn8s_settings.pkl" | tee \
# $logfile
#
# logfile="/home/nathan/histo-seg/semantic-pca/analysis_wsi/segnet_full/log.txt"
# ls /media/nathan/DATA/histo-seg-data/slide_for_testing_SPIE_2017/*svs | parallel --jobs 2 \
# "python /home/nathan/histo-seg/v2/core/histoseg.py --slide={} --settings=/home/nathan/histo-seg/v2/example/segnet_full_settings.pkl" | tee \
# $logfile
#
# logfile="/home/nathan/histo-seg/semantic-pca/analysis_wsi/segnet_basic_rotation/log.txt"
# ls /media/nathan/DATA/histo-seg-data/slide_for_testing_SPIE_2017/*svs | parallel --jobs 2 \
# "python /home/nathan/histo-seg/v2/core/histoseg.py --slide={} --settings=/home/nathan/histo-seg/v2/example/segnet_basic_rotation_settings.pkl" | tee \
# $logfile
#
# logfile="/home/nathan/histo-seg/semantic-pca/analysis_wsi/segnet_basic_crf_rotation/log.txt"
# ls /media/nathan/DATA/histo-seg-data/slide_for_testing_SPIE_2017/*svs | parallel --jobs 2 \
# "python /home/nathan/histo-seg/v2/core/histoseg.py --slide={} --settings=/home/nathan/histo-seg/v2/example/segnet_basic_crf_rotation_settings.pkl" | tee \
# $logfile
