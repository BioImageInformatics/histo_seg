'''
process.py
'''

import cv2
import numpy as np
import sys
import time

sys.path.insert(0, '.')
import data_utils

sys.path.insert(0, '/home/nathan/caffe-segnet-crf/python')
import caffe


def init_net(netproto, weights, gpumode=True):
    if gpumode:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    #/end if

    return caffe.Net(netproto, weights, caffe.TEST)
#/end init_net

'''
Assumes batchsize=1
need to take care of 3-channel and 1-channel
'''
def img_to_caffe(img):
    if len(img.shape) == 3:
        _,_,d = img.shape
        img = img.transpose((2,0,1))
        img = np.expand_dims(img, 0)
    else:
        d = 1
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 0)
    #/end if

    return img
#/end img_to_caffe

'''
For batchsize > 1
Also need to take care of 3-channel and 1-channel

What we need is the output to be: (h, w, n_class)
'''
def imgs_to_caffe_batch(imgs):
    pass

def run_net(net, img, layer='conv_classifier'):

    _ = net.forward(data=img_to_caffe(img))
    activ = np.squeeze(net.blobs[layer].data)

    ## Not sure about this one
    nd = activ.shape[0]
    activ_ = np.split(activ, nd, 0)
    activ_ = [np.squeeze(act) for act in activ_]
    activ_ = [np.expand_dims(act,2) for act in activ_]
    activ = np.dstack(activ_)

    return activ
#/end run_net


def process_svs(svs, prob_maps, coordinates, settings):
    # Check inputs
    overlaps = settings['overlaps']
    scales = settings['scales']
    weights = settings['weights']
    netproto = settings['deploy_proto']
    gpumode = settings['gpumode']
    cnnlayer = settings['cnnlayer']
    n_classes = settings['n_classes']
    prefetch = settings['prefetch']
    do_normalize = settings['do_normalize']
    do_normalize = False

    svs_info = {}
    svs_info = data_utils.pull_svs_stats(svs, svs_info)
    lvl20_index = svs_info['20x_lvl']
    mult_5x = svs_info['mult_5x']


    ## Loop over scales
    pmap_out = []
    for coords, scale, overlap, weight in zip(
        coordinates, scales, overlaps, weights):
        pmap_scale = np.zeros_like(prob_maps)
        net = init_net(netproto, weight, gpumode=gpumode)

        print 'Processing {}'.format(scale)

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

        ## A subset for speed
        # indices = np.random.choice(range(len(coords)), 500)
        # coords = [coords[index] for index in indices]
        # print 'Subsetted {} coordinates '.format(len(coords))

        failed_count = 0
        load_size = proc_size = settings['proc_size']
        load_size *= mult
        place_size = proc_size * place_mult
        place_size = int(place_size)

        ## Divide the set into n chunks
        if len(coords) < prefetch:
            coord_split = [coords]
            n_splits = 1
        else:
            n_splits = len(coords) / prefetch
            coord_split = np.array_split(coords, n_splits)
        #/end if

        print 'n splits: ', n_splits

        for nindx, coord_prefetch in enumerate(coord_split):
            ## tile preloading
            tiles = data_utils.preload_tiles(svs, coord_prefetch,
                size=(load_size, load_size), level=lvl20_index, normalize=do_normalize)
            tiles = [cv2.resize(tile, dsize=(proc_size, proc_size)) for tile in tiles]

            ## Processing here
            # tiles = [cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY) for tile in tiles]
            cnn_start = time.time()
            tiles = [run_net(net, tile, layer=cnnlayer) for tile in tiles]

            # Resize to fit
            tiles = [cv2.resize(tile, dsize=(place_size, place_size)) for tile in tiles]
            coord_prefetch = [(int(x * mult_5x), int(y * mult_5x)) for (x,y) in coord_prefetch]

            ## x, y are w.r.t. 20X
            for tile, (row, col) in zip(tiles, coord_prefetch):
                try:
                    pmap_scale[row:row+place_size, col:col+place_size, :] = tile
                except:
                    failed_count += 1
                #/end try
            #/end for
        #/end for

        print 'Failed: {}'.format(failed_count)

        pmap_out.append(pmap_scale)
    #/end for

    return pmap_out
#/end process_svs
