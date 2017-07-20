'''
tile.py
'''

import cv2
import os
import time
import numpy as np

import sys
sys.path.insert(0, '/home/nathan/histo-seg/v2/core')
import data_utils

from matplotlib import pyplot as plt

'''
We need something like this to deal with slides scanned at differing resolutions

In particular we have 20x and 40x slides.

The new way to think about it is the lowest dim will always be 5x.

[(40x), 20x, 10x, 5x]
'''
# def pull_svs_stats(svs, svs_info):
#     app_mag = svs.properties['aperio.AppMag']
#     level_dims = svs.level_dimensions
#
#     # Find 20X level:
#     if app_mag == '20':  # scanned @ 20X
#         svs_info['app_mag'] = '20'
#         svs_info['20x_lvl'] = 0
#         svs_info['20x_dim'] = level_dims[0][::-1]
#         svs_info['20x_downsample'] = svs.level_downsamples[0]
#     elif app_mag == '40':  # scanned @ 40X
#         svs_info['app_mag'] = '40'
#         svs_info['20x_lvl'] = 1
#         svs_info['20x_dim'] = level_dims[1][::-1]
#         svs_info['20x_downsample'] = svs.level_downsamples[1]
#     #/end if
#     svs_info['lvl0_dim'] = level_dims[0][::-1]
#     svs_info['low_downsample'] = svs.level_downsamples[-1]
#
#     return svs_info
# #/end pull_svs_stats


def whitespace(img, white_pt=210):
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #/end if
    bcg_level, img = cv2.threshold(img, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    img.dtype = np.bool
    return 1 - img
#/end whitespace


def get_process_map(masks):
    # Inverse the masks and take the union, by adding them together.
    inv = lambda x: 1 - x
    masks = [inv(mask) for mask in masks]
    n_masks = len(masks)
    if n_masks == 1:
        mask = masks[0]
    elif n_masks > 1:
        mask = np.add(masks)
    #/end if

    mask = mask == n_masks
    return mask
#/end get_process_map


def preprocessing(svs):
    img = data_utils.read_region(
        svs, 0, 0, svs.level_count - 1, svs.level_dimensions[-1], verbose=True)
    # s = 'PREPROCESS Successfully read image from {}\n'.format(
    #     kwargs['filename'])
    # logger(kwargs['reportfile'], s)

    # Boolean image of white areas
    whitemap = whitespace(img)

    # masks = [whitemap]
    # process_map = get_process_map(masks)
    # process_map = process_map.astype(np.uint8)
    process_map = whitemap.astype(np.uint8)

    return process_map
#/end preprocessing


def nrow_ncol(svs_info, tilesize, overlap):
    lvl20x = svs_info['20x_lvl']
    dim20x = svs_info['20x_dim']
    resize_factor = svs_info['20x_downsample']
    dims_top = svs_info['lvl0_dim']

    tile_top = int(
        dims_top[0] * tilesize / dim20x[0] / np.sqrt(resize_factor))
    overlap_top = int(overlap * np.sqrt(resize_factor))

    nrow = dims_top[1] / tile_top
    ncol = dims_top[0] / tile_top
    return tile_top, overlap_top, nrow, ncol
#/end nrow_ncol


def init_outputs(sample, n_classes):
    h, w = sample.shape[:2]
    prob_maps = []
    for k in range(n_classes):
        prob_maps.append(np.zeros(shape=(h,w), dtype=np.float32))
    #/end for

    prob_maps = np.dstack(prob_maps)

    return prob_maps
#/end init_outputs

'''
this thing has to preserve the overall shape of the foreground area
At this point foreground = 1 and background = 0

important to keep that convention
'''
def downsample_foreground(foreground, x, y):
    fg = cv2.resize(foreground, dsize=(x, y), interpolation=cv2.INTER_NEAREST)
    # fg =
    # fg = whitespace(fg)

    return fg



'''
return a list of coordinates we can use to
1) read from the svs file
2) place results into the prob_maps
'''
# def get_coordinates(svs, foreground, svs_info, settings):
def get_coordinates(svs, foreground, settings):
    scales = settings['scales']
    overlaps = settings['overlaps']

    svs_info = {}
    svs_info = data_utils.pull_svs_stats(svs, svs_info)
    lvl20_index = svs_info['20x_lvl']

    ## Dims are (row, col) = (y, x)
    lvl20_dims = svs_info['20x_dim']

    coordinates = []
    mults = []  # keep track of how to convert everything w.r.t. level 0
    # Get the multiplier to transform `scale` to 20X scale
    for scale, overlap in zip(scales, overlaps):
        proc_size = settings['proc_size']
        if scale == '20x':
            dims = svs.level_dimensions[-3][::-1]
            mult = 1
        elif scale == '10x':
            dims = svs.level_dimensions[-2][::-1]
            mult = 2
        elif scale == '5x':
            dims = svs.level_dimensions[-1][::-1]
            mult = 4
        #/end if

        print 'scale {}'.format(scale)
        print '\tDimensions: {}'.format(dims)
        lvl20size = proc_size * mult
        print '\tlvl20 Dimensions: {}'.format(lvl20_dims)
        print '\tTilesize w/r/t lvl20X: {}'.format(lvl20size)
        # tile_size = lvl20size - overlap
        # print '\tLattice initialized for 20X tilesize: {}'.format(tile_size)

        ## Get the lattice grid w/r/t level 0 and correctly sized tiles
        nrow, ncol = int(lvl20_dims[0]/lvl20size), int(lvl20_dims[1]/lvl20size)
        lst = [ (row, col) for row in range(1, nrow-2) for col in range(1, ncol-2) ]
        print '\tnrow: {} ncol: {}'.format(nrow, ncol)

        fg = downsample_foreground(foreground, ncol, nrow)
        print '\tFG downsampled to {}'.format(fg.shape)

        ## Transform from grid space to coordinate space w/r/t level 0
        lst = [ (int((row-1)*(lvl20size)), int((col-1)*(lvl20size)))
                for (row,col) in lst if fg[row,col] ]

        print '\tTransformed to {} - {}'.format(lst[0], lst[-1])
        print 'Returning {} coordinates'.format(len(lst))

        coordinates.append(lst)
        mults.append(mult)
    #/end for
    return coordinates, mults
#/end get_coordinates


def DEBUGGING_pull_tiles_from_coords(svs, coords, writeto, size=256, level=-1):
    assert os.path.exists(writeto), '{} does not exist'.format(writeto)

    # print '{} coordinates'.format(len(coords))
    size_mult = int(np.sqrt(svs.level_downsamples[level]))

    indices = range(len(coords))
    indices = np.random.choice(indices, 100)

    for i, index in enumerate(indices):
        x, y = coords[index]
        img = data_utils.read_region(svs, x, y, 0, size=(size*size_mult, size*size_mult))
        img = cv2.resize(img, dsize=(size,size))
        img_name = os.path.join(writeto, '{}x{}x{}.jpg'.format(x,y,size))
        cv2.imwrite(filename=img_name, img=img)
    #/end for
#/end DEBUGGING_pull_tiles_from_coords


def tile_svs(svs, settings):
    foreground = preprocessing(svs)

    # probability images are at 5x
    prob_maps = init_outputs(foreground, settings['n_classes'])
    print 'Scanning for foreground tiles'
    # coordinates, coordinates_low = get_coordinates(svs, foreground, svs_info, settings)
    coordinates, _ = get_coordinates(svs, foreground, settings)
    print 'Found {} foreground candidates'.format(len(coordinates[0]))

    return coordinates, prob_maps, foreground
#/end tile_svs
