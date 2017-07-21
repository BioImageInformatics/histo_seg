#/bin/bash

# for svs in $( ls /home/nathan/data/pca_wsi/SPIE_TEST/*svs ); do
#   echo $svs
# done

ls /home/nathan/data/pca_wsi/*svs | parallel --jobs 2 \
"python /home/nathan/histo-seg/v2/core/histoseg.py --slide={} --settings=/home/nathan/histo-seg/v2/example/pca_settings.pkl" | tee \
/home/nathan/histo-seg/semantic-pca/analysis_wsi_log.txt
