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
import shutil

module_dir, module_name = os.path.split(__file__)
sys.path.insert(0, module_dir)
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
    svs_info['20x_to_5x'] = 1/16.
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
    tiles = [read_region(svs, x, y, level, size= size)
             for (y,x) in coords]

    if normalize:
        tiles = [cnorm.normalize(tile) for tile in tiles]
    # tiles = [cv2.resize(tile, dsize=(size,size)) for tile in tiles]

    if as_ndarray:
        tiles = [np.expand_dims(tile, axis=0) for tile in tiles]
        tiles = np.concatenate(tiles, axis=0)
    #/end if

    return tiles
#/end preload_tiles


''' Just a helper '''
def read_low_level(svs, verbose=False):
    return read_region(svs, 0, 0, svs.level_count - 1,
        svs.level_dimensions[-1], verbose=verbose)
#/end read_low_level


''' copy file to ramdisk '''
def transfer_to_ramdisk(filename, destination):
    newname = os.path.basename(filename)
    newname = os.path.join(destination, newname)

    try:
        shutil.copyfile(filename, newname)
        print 'Copied {} to {}'.format(filename, newname)
    except:
        print 'Failed to transfer {} to {}'.format(filename, newname)
        return 0

    return newname
#/end transfer_to_ramdisk


def delete_from_ramdisk(filename):
    try:
        os.remove(filename)
    except:
        return 0
    #/end try
#/end transfer_to_ramdisk

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
            mult = 1
            # for index in np.unique(img):
            #     filename_ = os.path.join(output_dir, svsbase+'_{}_'.format(index)+filename+ext)
            #     cv2.imwrite(filename_, (img == index).astype(np.uint8)*255)

        elif filename == 'probability':
            ext = '.jpg'
            mult = 255/img.max()
        else:
            ext = '.jpg'
            mult = 1
        #/end if

        filename_ = os.path.join(output_dir, svsbase+'_'+filename+ext)
        print 'Saving {}'.format(filename_)
        cv2.imwrite(filename_, img*mult)
    #/end for
#/end save_result
