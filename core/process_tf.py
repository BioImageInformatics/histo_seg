'''
process.py
'''

import cv2
import numpy as np
import sys
import time
import os
import random

module_dir, module_name = os.path.split(__file__)
sys.path.insert(0, module_dir)
import data_utils
import colorNormalization as cnorm


def bayesian_inference(net, image, samples=25, keep_prob=0.5):
    y_hat = net.inference(image, keep_prob=keep_prob)
    y_hat = np.expand_dims(y_hat, -1) ## (h,w,d,1)

    for _ in xrange(samples):
        y_hat_p += net.inference(image, keep_prob=keep_prob)
        y_hat = np.concatenate([y_hat, np.expand_dims(y_hat_p, -1)], -1)

    y_bar = np.mean(y_hat, axis=-1)
    y_hat_var = np.var(y_hat, axis=-1)
    return y_bar, y_hat_var



def place_tiles_into(tiles, in_img, processed, place_size, coord_prefetch, overlap, mult_5x):
    tiles = [np.squeeze(tile) for tile in tiles]
    tiles = [cv2.resize(tile, dsize=(place_size, place_size)) for tile in tiles]
    coord_prefetch = [(int(x * mult_5x), int(y * mult_5x)) for (x,y) in coord_prefetch]

    ## x, y are w.r.t. 20X
    if overlap < 1 and overlap > 0:
        overlap = load_size * overlap
    ovp = int(overlap * mult_5x)
    inner = [ovp, place_size-ovp]
    in_out = np.zeros((place_size, place_size), dtype=np.bool)
    in_out[inner[0]:inner[1], inner[0]:inner[1]] = True
    for tile, (row, col) in zip(tiles, coord_prefetch):
        # try:
        placeholder = in_img[row:row+place_size, col:col+place_size, :]
        processed_pl = processed[row:row+place_size, col:col+place_size]
        if (processed_pl).sum() > 0:
            ## we've already placed some of this tile
            placeholder[in_out] = tile[in_out]
            tile_out = tile[in_out==0]

            ## Take a dirty average
            placeholder[in_out==0] += tile_out
            placeholder[in_out==0] /= 2
            in_img[row:row+place_size, col:col+place_size, :] = placeholder
        else:
            ## We haven't placed any part of this tile; place in the whole thing.
            in_img[row:row+place_size, col:col+place_size, :] = tile
        processed[row:row+place_size, col:col+place_size] = True

        return np.copy(in_img), np.copy(processed)


def get_load_place_proc_size(scale, settings):
        if scale == '20x':
            mult = 1
            place_mult = 0.25**2
        elif scale == '10x':
            mult = 2
            place_mult = 1/8.
        elif scale == '5x':
            mult = 4
            place_mult = 0.25

        load_size = proc_size = settings['proc_size']
        load_size *= mult
        place_size = int(proc_size * place_mult)

        return load_size, place_size, proc_size

def process_svs(svs, prob_maps, coordinates, net, settings):
    # Check inputs
    overlap = settings['overlap']
    scales = settings['scales']
    # weights = settings['weights']
    # netproto = settings['deploy_proto']
    gpumode = settings['gpumode']
    # cnnlayer = settings['cnnlayer']
    n_classes = settings['n_classes']
    prefetch = settings['prefetch']
    do_normalize = settings['do_normalize']
    debug_mode = settings['DEBUGGING']

    ## Set the processing funciton up front to avoid if-else hell
    # if bayesian:
    #     print 'Setting up for bayesian inference mode with {} samples'.format(samples)
    #     process_fn = lambda x: net.bayesian_inference(x, samples=samples)
    # else:
    print 'Using normal forward pass, set keep_prob=1.0'
    process_fn = lambda x: net.inference(x, keep_prob=1.0)

    svs_info = {}
    svs_info = data_utils.pull_svs_stats(svs, svs_info)
    lvl20_index = svs_info['20x_lvl']
    mult_5x = svs_info['20x_to_5x']

    ## Loop over scales
    pmap_out = []
    for coords, scale in zip(coordinates, scales):
        pmap_scale = np.copy(prob_maps)  ## use this line to enforce default class
        h,w = pmap_scale.shape[:2]
        processed = np.zeros((h,w), dtype=np.bool)

        print 'Processing {}'.format(scale)
        print 'Using {} tile coordinates'.format(len(coords))

        load_size, place_size, proc_size = get_load_place_proc_size(scale, settings)
        failed_count = 0

        ## Leftover from debugging
        # print 'Shuffling coordinates'
        # random.shuffle(coords)
        # indices = np.random.choice(range(len(coords)), 250)
        # coords = [coords[index] for index in indices]
        # print 'Subsetted {} coordinates '.format(len(coords))

        ## Divide the set into n chunks
        if len(coords) < prefetch:
            print 'Coordinates ({}) < prefetch ({})'.format(
                len(coords), prefetch )
            coord_split = [coords]
            n_splits = 1
        else:
            n_splits = len(coords) / prefetch
            coord_split = np.array_split(coords, n_splits)

        for nindx, coord_prefetch in enumerate(coord_split):
            print '[{:02d}/{:02d}]'.format(nindx+1, n_splits),

            preload_start = time.time()
            tiles = data_utils.preload_tiles(svs, coord_prefetch,
                    size=(load_size, load_size), level=lvl20_index)
            tiles = [cv2.resize(tile, dsize=(proc_size, proc_size)) for tile in tiles]
            preload_delta_t = time.time() - preload_start
            print '{} Tiles preloaded in {:03.3f}s'.format(
                len(tiles), preload_delta_t),

            if do_normalize:
                norm_start = time.time()
                tiles = [cnorm.normalize(tile) for tile in tiles]
                norm_delta_t = time.time() - norm_start
                print 'Normalizing done in {:03.3f}s'.format(norm_delta_t),
            #/end if

            ## Processing here
            # tiles = [cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY) for tile in tiles]
            cnn_start = time.time()
            tiles = [np.expand_dims(tile, 0) for tile in tiles]
            tiles = [tile*(2/255.0)-1 for tile in tiles] ## Recenter to [-1,1] for SELU activations
            tiles = [process_fn(tile) for tile in tiles]
            cnn_delta_t = time.time() - cnn_start
            print 'CNN finished in {:03.3f}s'.format(cnn_delta_t)

            # Resize to fit
            placing_start = time.time()

            #def place_tiles_into(tiles, in_img, processed, place_size, coord_prefetch, overlap, mult_5x):
            #pmap_scale, processed = place_tiles_into(tiles, pmap_scale, processed, place_size, coord_prefetch, overlap, mult_5x)

            tiles = [np.squeeze(tile) for tile in tiles]
            tiles = [cv2.resize(tile, dsize=(place_size, place_size)) for tile in tiles]
            coord_prefetch = [(int(x * mult_5x), int(y * mult_5x)) for (x,y) in coord_prefetch]
#
            ## x, y are w.r.t. 20X
            if overlap < 1 and overlap > 0:
                overlap = load_size * overlap
            ovp = int(overlap * mult_5x)
            inner = [ovp, place_size-ovp]
            in_out = np.zeros((place_size, place_size), dtype=np.bool)
            in_out[inner[0]:inner[1], inner[0]:inner[1]] = True
            for tile, (row, col) in zip(tiles, coord_prefetch):
                # try:
                placeholder = pmap_scale[row:row+place_size, col:col+place_size, :]
                processed_pl = processed[row:row+place_size, col:col+place_size]
                if (processed_pl).sum() > 0:
                    ## we've already placed some of this tile
                    placeholder[in_out] = tile[in_out]
                    tile_out = tile[in_out==0]
#
                    ## Take a dirty average
                    placeholder[in_out==0] += tile_out
                    placeholder[in_out==0] /= 2
                    pmap_scale[row:row+place_size, col:col+place_size, :] = placeholder
                else:
                    ## We haven't placed any part of this tile; place in the whole thing.
                    pmap_scale[row:row+place_size, col:col+place_size, :] = tile
                processed[row:row+place_size, col:col+place_size] = True

            placing_delta_t = time.time() - placing_start
        print 'Done placing tiles in {}s'.format(placing_delta_t)
        print 'Failed: {}'.format(failed_count)

        pmap_out.append(pmap_scale)

    return pmap_out
