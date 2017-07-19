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

'''
We need something like this to deal with slides scanned at differing resolutions

In particular we have 20x and 40x slides.

The new way to think about it is the lowest dim will always be 5x.

[(40x), 20x, 10x, 5x]
'''
def pull_svs_stats(svs, svs_info):
    app_mag = svs.properties['aperio.AppMag']
    level_dims = svs.level_dimensions

    # Find 20X level:
    if app_mag == '20':  # scanned @ 20X
        svs_info['app_mag'] = '20'
        svs_info['20x_lvl'] = 0
        svs_info['20x_dim'] = level_dims[0][::-1]
        svs_info['20x_downsample'] = svs.level_downsamples[0]
    elif app_mag == '40':  # scanned @ 40X
        svs_info['app_mag'] = '40'
        svs_info['20x_lvl'] = 1
        svs_info['20x_dim'] = level_dims[1][::-1]
        svs_info['20x_downsample'] = svs.level_downsamples[1]
    #/end if
    svs_info['lvl0_dim'] = level_dims[0][::-1]
    svs_info['low_downsample'] = svs.level_downsamples[-1]

    return svs_info
#/end pull_svs_stats


def whitespace(img, white_pt=210):
    background = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bcg_level, background = cv2.threshold(background, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    background = cv2.morphologyEx(background, cv2.MORPH_OPEN, kernel)

    background.dtype = np.bool
    return background
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
    img = data_utils.read_region(svs, 0, 0, -1, svs.level_dimensions[-1])
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
    outputs = []
    for k in range(n_classes):
        outputs.append(np.zeros(shape=(h,w), dtype=np.float32))
    #/end for

    return outputs
#/end init_outputs


def downsample_foreground(foreground, nrow, ncol):
    fg = cv2.resize(foreground, dsize=(nrow, ncol), interpolation=cv2.INTER_LINEAR)
    fg = whitespace(fg)

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


    coordinates = []
    mults = []  # keep track of how to convert everything w.r.t. level 0
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

        nrow, ncol = dims[0]/int(proc_size/mult), dims[1]/int(proc_size/mult)
        print '{}: dims:{} nrow: {}, ncol: {}'.format(scale, dims, nrow, ncol)
        lst = [(x,y)
                for x in range(1, ncol-1)
                for y in range(1, nrow-1)]

        print 'Original list: ', len(lst)

        fg = cv2.resize(foreground, dsize=(nrow, ncol), interpolation=cv2.INTER_NEAREST)
        proc_size -= 2*int(overlap / mult)

        print 'Overlap: ', overlap/mult
        print 'Proc size to use: ', proc_size

        lst = [(int(x*proc_size*mult)-overlap/mult, int(y*proc_size*mult)-overlap/mult)
               for (x, y) in lst if fg[x,y]]

        print 'filtered list: ', len(lst)
        coordinates.append(lst)
        mults.append(mult)
    #/end for
    return coordinates, mults
#/end get_coordinates


def DEBUGGING_pull_tiles_from_coords(svs, coords, writeto, size=256, level=-1):
    assert os.path.exists(writeto), '{} does not exist'.format(writeto)

    print '{} coordinates'.format(len(coords))
    size_mult = int(np.sqrt(svs.level_downsamples[level]))

    indices = range(len(coords))
    indices = np.random.choice(indices, 100)

    for i, index in enumerate(indices):
        x, y = coords[index]
        img = data_utils.read_region(svs, x, y, 0, size=(size*size_mult, size*size_mult))
        # img = data_utils.read_region(
        #     svs, start=(x1,y1), level=level, dims=size)
        img = cv2.resize(img, dsize=(size,size))
        img_name = os.path.join(writeto, '{}x{}x{}.jpg'.format(x,y,size))
        cv2.imwrite(filename=img_name, img=img)
        print 'region {}, x:{}, y:{}, shape:{}'.format(
            i, x, y, img.shape)
    #/end for
#/end DEBUGGING_pull_tiles_from_coords


def tile_svs(svs, settings):
    # pull basic info from svs
    # svs_info = {}
    # svs_info = pull_svs_stats(svs, svs_info)

    # find foreground area
    # foreground is at 5x
    foreground = preprocessing(svs)

    # probability images are at 5x
    prob_maps = init_outputs(foreground, settings['n_classes'])
    print 'Scanning for foreground tiles'
    # coordinates, coordinates_low = get_coordinates(svs, foreground, svs_info, settings)
    coordinates, mults = get_coordinates(svs, foreground, settings)
    print 'Found {} foreground candidates'.format(len(coordinates[0]))

    return coordinates, prob_maps, mults
#/end tile_svs
