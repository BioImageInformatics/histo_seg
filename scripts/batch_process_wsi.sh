#/bin/bash

set -e

# for svs in $( ls /home/nathan/data/pca_wsi/SPIE_TEST/*svs ); do
#   echo $svs
# done

logfile="/home/nathan/histo-seg/semantic-pca/analysis_wsi/segnet_basic_crf/log.txt"

ls /media/nathan/DATA/histo-seg-data/slide_for_testing_SPIE_2017/*svs | parallel --jobs 2 \
"python /home/nathan/histo-seg/v2/core/histoseg.py --slide={} --settings=/home/nathan/histo-seg/v2/example/pca_settings.pkl" | tee \
$logfile
