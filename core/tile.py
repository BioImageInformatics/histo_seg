'''
tile.py
'''

import cv2
import os
import time
import numpy as np

import sys
module_dir, module_name = os.path.split(__file__)
sys.path.insert(0, module_dir)
import data_utils

from matplotlib import pyplot as plt

'''
We need something like this to deal with slides scanned at differing resolutions

In particular we have 20x and 40x slides.

The new way to think about it is the lowest dim will always be 5x.

[(40x), 20x, 10x, 5x]
'''

def whitespace(img, mode='Otsu', white_pt=210):
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #/end if

    if mode=='Ostu':
        bcg_level, img = cv2.threshold(img, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return img > 0
    elif mode=='thresh':
        foreground = (img < white_pt).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

        return foreground > 0
    else:
        raise ValueError('tile::whitespace mode must be "Otsu" or "thresh"')
    #/end if
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


'''
https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
'''
def imfill(img):
    if img.dtype == 'bool':
        img = img.astype(np.uint8)
    #/end if

    ## Old way:
    # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

    # open cv contours
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ## TODO hard code area threshold
    hulls = [cv2.convexHull(cnt) for cnt in cnts if cv2.contourArea(cnt) > 2000]
    img2 = np.zeros_like(img)
    cv2.drawContours(img2, hulls, -1, (1), -1)

    return img2 > 0
#/end imfill


def preprocessing(svs):
    img = data_utils.read_low_level(svs)

    # Boolean image of white areas
    whitemap = whitespace(img, mode='thresh')
    whitemap_filled = imfill(whitemap)


    ## Really shouldn't need this
    if whitemap_filled.dtype == 'bool':
        process_map = whitemap_filled.astype(np.uint8)
    elif whitemap_filled.dtype == 'uint8':
        process_map = whitemap_filled
    #/end if

    return process_map, whitemap
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

'''
Should return float32 3D array (h,w,n_classes)

Trying to set a default class here.. it's not working.
'''
def init_outputs(foreground, n_classes, default_class):
    h, w = foreground.shape[:2]
    prob_maps = []
    for k in range(n_classes):
        # prob_maps.append(np.zeros(shape=(h,w), dtype=np.float32))
        if k == default_class:
            prob_maps.append(np.ones(shape=(h,w), dtype=np.float32))
        else:
            prob_maps.append(np.zeros(shape=(h,w), dtype=np.float32))
        #/end if
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
    overlap = settings['overlap']

    svs_info = {}
    svs_info = data_utils.pull_svs_stats(svs, svs_info)
    lvl20_index = svs_info['20x_lvl']

    ## Dims are (row, col) = (y, x)
    lvl20_dims = svs_info['20x_dim']

    coordinates = []
    mults = []  # keep track of how to convert everything w.r.t. level 0
    # Get the multiplier to transform `scale` to 20X scale
    for scale in scales:
        proc_size = settings['proc_size']
        if scale == '20x':
            mult = 1
        elif scale == '10x':
            mult = 2
        elif scale == '5x':
            mult = 4
        #/end if

        lvl20size = proc_size * mult

        ## Get the lattice grid w/r/t level 0 and correctly sized tiles
        nrow, ncol = int(lvl20_dims[0]/(lvl20size-overlap)), int(lvl20_dims[1]/(lvl20size-overlap))
        lst = [ (row, col) for row in range(1, nrow-2) for col in range(1, ncol-2) ]
                                            # print '\tnrow: {} ncol: {}'.format(nrow, ncol)

        fg = downsample_foreground(foreground, ncol, nrow)
                                            # print '\tFG downsampled to {}'.format(fg.shape)

        ## Transform from grid space to coordinate space w/r/t level 0
        lst = [ (int((row-1)*(lvl20size-overlap)), int((col-1)*(lvl20size-overlap)))
                for (row,col) in lst if fg[row,col] ]

        coordinates.append(lst)
        mults.append(mult)
    #/end for
    return coordinates, mults
#/end get_coordinates


'''
take in an svs file and settings,

return a list of coordinates according to the tile size, and overlap
'''
def tile_svs(svs, settings):
    foreground, original_foreground = preprocessing(svs)
    # background = cv2.bitwise_not(foreground)
    background = 1 - foreground

    # probability images are at 5x
    prob_maps = init_outputs(foreground, settings['n_classes'], settings['replace_value'])
    print 'Scanning for foreground tiles'
    # coordinates, coordinates_low = get_coordinates(svs, foreground, svs_info, settings)
    coordinates, _ = get_coordinates(svs, foreground, settings)
    print 'Found {} foreground candidates'.format(len(coordinates[0]))

    return coordinates, prob_maps, background, (1-original_foreground)
#/end tile_svs
