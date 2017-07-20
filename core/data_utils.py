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

def pull_svs_stats(svs, svs_info):
    app_mag = svs.properties['aperio.AppMag']
    level_dims = svs.level_dimensions

    # Find 20X level:
    if app_mag == '20':  # scanned @ 20X
        svs_info['app_mag'] = '20'
        svs_info['20x_lvl'] = 0
        svs_info['20x_dim'] = level_dims[0][::-1]
        # svs_info['20x_dim'] = level_dims[0]
        svs_info['20x_downsample'] = svs.level_downsamples[0]
        svs_info['mult_5x'] = 1/16.
    elif app_mag == '40':  # scanned @ 40X
        print 'WARNING CHECK 40X NUMBERS'
        svs_info['app_mag'] = '40'
        svs_info['20x_lvl'] = 1
        svs_info['20x_dim'] = level_dims[1][::-1]
        # svs_info['20x_dim'] = level_dims[1]
        svs_info['20x_downsample'] = svs.level_downsamples[1]
        svs_info['mult_5x'] = 1/64.
    #/end if
    # svs_info['lvl0_dim'] = level_dims[0][::-1]
    svs_info['lvl0_dim'] = level_dims[0]
    svs_info['low_downsample'] = svs.level_downsamples[-1]

    return svs_info
#/end pull_svs_stats

def read_region(svs, x, y, level, size, verbose=False):
    # Utility function because openslide loads as RGBA
    if verbose:
        print 'Reading SVS: ({},{}), LEVEL {}, SIZE={}'.format(
            x,y,level,size)
    #/end if

    ## Check if region is out of range for the requested level

    img = svs.read_region((x,y), level, size)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
#/end read_region

'''
Load up a bunch of tiles as an ndarray ??
'''
def preload_tiles(svs, coords, level, size, as_ndarray=False):
    # tiles = []
    # # size_mult = int(np.sqrt(svs.level_downsamples[level]))
    # for i, (x, y) in enumerate(coords):
    #     tiles.append(
    #         read_region(svs, x, y, level, size=size))
    # #/end for
    tiles = [read_region(svs, x, y, level, size= size)
             for (y,x) in coords]

    # tiles = [cv2.resize(tile, dsize=(size,size)) for tile in tiles]

    if as_ndarray:
        tiles = [np.expand_dims(tile, axis=0) for tile in tiles]
        tiles = np.concatenate(tiles, axis=0)
    #/end if

    return tiles
#/end preload_tiles
