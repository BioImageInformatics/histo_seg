#/bin/bash

set -e

# for svs in $( ls /home/nathan/data/pca_wsi/SPIE_TEST/*svs ); do
#   echo $svs
# done

logfile="/home/nathan/histo-seg/semantic-pca/analysis_wsi/segnet_basic_wholeset/log.txt"
ls /media/nathan/DATA/histo-seg-data/slide_for_testing_SPIE_2017/*svs | parallel --jobs 3 \
"python /home/nathan/histo-seg/v2/core/histoseg.py --slide={} --settings=/home/nathan/histo-seg/v2/example/segnet_basic_whole_settings.pkl" | tee \
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
