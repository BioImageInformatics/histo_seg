'''

data_utils.py

utility functions for read/write

'''
from openslide import OpenSlide
import cv2
import numpy as np
import time
import sys
import os
sys.path.insert(0, '/home/nathan/histo-seg/v2/core')
import colorNormalization as cnorm

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
        svs_info['20x_downsample'] = svs.level_downsamples[0]
        svs_info['mult_5x'] = 1/16.
    elif app_mag == '40':  # scanned @ 40X
        print 'WARNING CHECK 40X NUMBERS'
        svs_info['app_mag'] = '40'
        svs_info['20x_lvl'] = 1
        svs_info['20x_dim'] = level_dims[1][::-1]
        svs_info['20x_downsample'] = svs.level_downsamples[1]
        svs_info['mult_5x'] = 1/64.
    #/end if
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
def preload_tiles(svs, coords, level, size, as_ndarray=False, normalize=True):
    time_start = time.time()
    tiles = [read_region(svs, x, y, level, size= size)
             for (y,x) in coords]

    if normalize:
        tiles = [cnorm.normalize(tile) for tile in tiles]
    # tiles = [cv2.resize(tile, dsize=(size,size)) for tile in tiles]

    print 'Loaded {} tiles in {:3.3f}s'.format(len(tiles), time.time() - time_start)
    if as_ndarray:
        tiles = [np.expand_dims(tile, axis=0) for tile in tiles]
        tiles = np.concatenate(tiles, axis=0)
    #/end if

    return tiles
#/end preload_tiles




def read_low_level(svs, verbose=False):
    return read_region(svs, 0, 0, svs.level_count - 1,
        svs.level_dimensions[-1], verbose=verbose)
#/end read_low_level



'''
Return the basename
'''
def svs_name(pathname):
    base = os.path.basename(pathname)
    a, ext = os.path.splitext(base)
    return a
#/end svs_name


'''
utility to save results
'''
def save_result(imgs, svsbase, settings):
    output_dir = settings['output_dir']
    output_filenames = settings['output_filenames']
    n_classes = settings['n_classes']

    assert len(imgs) == len(output_filenames)

    for img, filename in zip(imgs, output_filenames):
        if filename == 'argmax':
            ext = '.png'
            mult = 255/n_classes
            # mult = 1
        elif filename == 'probability':
            ext = '.jpg'
            mult = 255
        else:
            ext = '.jpg'
            mult = 1
        #/end if

        filename_ = os.path.join(output_dir, svsbase+'_'+filename+ext)
        print 'Saving {}'.format(filename_)
        cv2.imwrite(filename_, img*mult)
    #/end for
#/end save_result
