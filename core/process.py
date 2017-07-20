'''
process.py
'''

import cv2
import numpy as np
import sys

sys.path.insert(0, '.')
import data_utils


def process_svs(prob_maps, coordinates, settings):
    # Check inputs
    return 0


def DEBUGGING_fill_prob_maps(svs, prob_maps, coordinates, settings):
    overlaps = settings['overlaps']
    scales = settings['scales']

    svs_info = {}
    svs_info = data_utils.pull_svs_stats(svs, svs_info)
    lvl20_index = svs_info['20x_lvl']
    mult_5x = svs_info['mult_5x']

    # coordinates are w.r.t level 0
    # pmap is num_levels - 1
    # pmap is (h, w, num_classes)
    pmap = np.squeeze(prob_maps[:,:,0]) # for debugging
    pmap_out = []
    for coords, scale, overlap in zip(coordinates, scales, overlaps):
        if scale == '20x':
            dims = svs.level_dimensions[-3][::-1]
            mult = 1
            place_mult = 0.25**2
        elif scale == '10x':
            dims = svs.level_dimensions[-2][::-1]
            mult = 2
            place_mult = 1/8.
        elif scale == '5x':
            dims = svs.level_dimensions[-1][::-1]
            mult = 4
            place_mult = 0.25
        #/end if

        pmap_scale = np.zeros_like(pmap)
        load_size = settings['proc_size']
        proc_size = settings['proc_size']

        print 'FILLING scale {}'.format(scale)
        print 'proc_size: {}'.format(proc_size)
        load_size *= mult
        # load_size -= 2*overlap
        print 'LOADING tiles size {} from level {}'.format(load_size, lvl20_index)

        ## A subset for speed
        # indices = np.random.choice(range(len(coords)), 1000)
        # coords = [coords[index] for index in indices]
        # print 'Subsetted {} coordinates '.format(len(coords))

        ## tile preloading
        tiles = data_utils.preload_tiles(svs, coords, size=(load_size, load_size), level=lvl20_index)
        print '\tloaded {} tiles ({})'.format(len(tiles), tiles[0].shape)
        print '\tresizing to ({})'.format(proc_size)
        tiles = [cv2.resize(tile, dsize=(proc_size, proc_size)) for tile in tiles]

        ## Processing here
        tiles = [cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY) for tile in tiles]
        # tiles = [np.ones(shape=(256,256))*np.random.randint(255) for _ in coords]

        # Resize to fit
        print 'PLACING'
        print '\ttarget image is {}'.format(pmap_scale.shape)
        print '\tcoords start at {}, {}'.format(coords[0][0]*mult_5x, coords[0][1]*mult_5x)
        print '\t..and end at {}, {}'.format(coords[-1][0]*mult_5x, coords[-1][1]*mult_5x)
        # print '\tmultiplier is {}'.format(mult_5x)
        proc_size *= place_mult
        proc_size = int(proc_size)
        print '\tRESIZING tiles to {}'.format(proc_size)
        tiles = [cv2.resize(tile, dsize=(proc_size, proc_size)) for tile in tiles]

        print 'Chaning coordinates back to 5x'
        coords = [(int(x * mult_5x), int(y * mult_5x)) for (x,y) in coords]

        print 'Searching for extreme coord values:'
        max_x = max_y = 0
        for (x,y) in coords:
            if x > max_x: max_x = x
            if y > max_y: max_y = y
        print '\tMax x: {}'.format(max_x)
        print '\tMax y: {}'.format(max_y)

        ## x, y are w.r.t. 20X
        failed_count = 0
        for tile, (row, col) in zip(tiles, coords):
            # row,col = int(row * mult_5x), int(col * mult_5x)
            try:
                pmap_scale[row:row+proc_size, col:col+proc_size] = tile
                # print 'placed {} {}'.format(x, y)
            except:
                # pass
                failed_count += 1
                print '[{} {}, ({})]'.format(row, col, tile.shape), #/end try
                # continue
        #/end for
        print '\tFAILED COUNT = {}'.format(failed_count)
        pmap_out.append(pmap_scale)
    #/end for

    return pmap_out
