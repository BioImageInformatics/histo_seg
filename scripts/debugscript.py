import sys
import os
import cv2
import time
from openslide import OpenSlide
import numpy as np
import cPickle as pickle

from matplotlib import pyplot as plt

settingsfile = '/Users/nathaning/_projects/histo-seg-v2/example/pca_settings.pkl'
core_dir = '/Users/nathaning/_projects/histo-seg-v2/core'
debug_dir = '/Users/nathaning/_projects/histo-seg-v2/data'

svs = OpenSlide('/Users/nathaning/Dropbox/projects/semantic_pca/data/1305462.svs')
with open(settingsfile, 'r') as f:
    settings = pickle.load(f)

sys.path.insert(0, core_dir)
import tile
import data_utils
import process

time_start = time.time()
coordinates, prob_maps, foreground = tile.tile_svs(svs, settings)

##
print 'Printing some example coords:'
print '10x'
for (x,y) in coordinates[0][:10]:
    print x,y
print '5x'
for (x,y) in coordinates[1][:10]:
    print x,y

# print 'Searching for extreme coord values:'
# max_x = max_y = 0
# for (x,y) in coordinates[0]:
#     if x > max_x: max_x = x
#     if y > max_y: max_y = y
# print 'Scale 10x'
# print '\tMax x: {}'.format(max_x)
# print '\tMax y: {}'.format(max_y)
# max_x = max_y = 0
# for (x,y) in coordinates[0]:
#     if x > max_x: max_x = x
#     if y > max_y: max_y = y
# print 'Scale 5x'
# print '\tMax x: {}'.format(max_x)
# print '\tMax y: {}'.format(max_y)
# x, y = zip(*coordinates[0])
# plt.figure();plt.scatter(x, y)
# x, y = zip(*coordinates[1])
# plt.figure();plt.scatter(x, y)
#
# plt.matshow(foreground)
# plt.show()

cv2.imwrite(os.path.join(debug_dir, '_foreground.jpg'), foreground*255)

# tile.DEBUGGING_pull_tiles_from_coords(svs, coordinates[1], debug_dir, level=-1)
# tile.DEBUGGING_pull_tiles_from_coords(svs, coordinates[0], debug_dir, level=-2)

pmap = process.DEBUGGING_fill_prob_maps(svs, prob_maps, coordinates, settings)

for i, pm in enumerate(pmap):
    cv2.imwrite(os.path.join(debug_dir, '_pmap{}.jpg'.format(i)), pm)


elapsed = time.time() - time_start
print 'Elapsed: {}s'.format(elapsed)
