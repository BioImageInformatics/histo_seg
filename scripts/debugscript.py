import sys
import os
import cv2
import time
from openslide import OpenSlide
import numpy as np
import cPickle as pickle

from matplotlib import pyplot as plt

# settingsfile = '/Users/nathaning/_projects/histo-seg-v2/example/pca_settings.pkl'
# core_dir = '/Users/nathaning/_projects/histo-seg-v2/core'
# debug_dir = '/Users/nathaning/_projects/histo-seg-v2/data'

settingsfile = '/home/nathan/histo-seg/v2/example/pca_settings.pkl'
core_dir = '/home/nathan/histo-seg/v2/core'
debug_dir = '/dev/shm/db'

svs = OpenSlide('/dev/shm/1305400.svs')
with open(settingsfile, 'r') as f:
    settings = pickle.load(f)

sys.path.insert(0, core_dir)
import tile
import data_utils
import process

time_start = time.time()
coordinates, prob_maps, foreground = tile.tile_svs(svs, settings)

cv2.imwrite(os.path.join(debug_dir, '_foreground.png'), foreground*255)

pmaps = process.process_svs(svs, prob_maps, coordinates, settings)

for i, pm in enumerate(pmaps):
    pm_ = pm[:,:,:3]
    print 'map: ', pm.shape, pm.min(), pm.max()
    cv2.imwrite(os.path.join(debug_dir, '_pmap{}.jpg'.format(i)), pm_*255)

    pm_max = np.argmax(pm, axis=2)
    cv2.imwrite(os.path.join(debug_dir, '_pmax{}.jpg'.format(i)), pm_max*(255)/4)


elapsed = time.time() - time_start
print 'Elapsed: {}s'.format(elapsed)
