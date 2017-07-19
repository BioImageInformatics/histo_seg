'''

data_utils.py

utility functions for read/write

'''
from openslide import OpenSlide
import cv2
import numpy as np
import time

def open_slide(slide):
    return OpenSlide(slide)
#/end retrn_slide

def read_region(wsi, x, y, level, size):
    # Utility function because openslide loads as RGBA
    if level < 0:
        level = wsi.level_count + level -1

    img = wsi.read_region((x,y), level, size)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
#/end read_region

'''
Load up a bunch of tiles as an ndarray ??
'''
def preload_tiles(svs, coords, writeto, size=256, level=-1):
    assert os.path.exists(writeto), '{} does not exist'.format(writeto)

    tiles = []
    size_mult = int(np.sqrt(svs.level_downsamples[level]))

    for i, (x, y) in enumerate(coords):
        img = data_utils.read_region(svs, x, y, 0, size=(size*size_mult, size*size_mult))
        # img = data_utils.read_region(
        #     svs, start=(x1,y1), level=level, dims=size)
        img = cv2.resize(img, dsize=(size,size))
        img_name = os.path.join(writeto, '{}x{}x{}.jpg'.format(x,y,size))
    #/end for
#/end preload_tiles
