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

def init_net(netproto, weights, caffe_root, gpumode=True):
    sys.path.insert(0, caffe_root)
    try:
        import caffe
    except:
        print 'ERROR: Failed to load pyCaffe from {}'.format(caffe_root)
    #/end try

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
        # imagetiletmp_r1 = np.rollaxis(imagetiletmp_r1, -1, 0)
        # img = np.rollaxis(img, -1, 0)
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
#/end imgs_to_caffe_batch



''' This function instead of np.rollaxis or similar '''
def activations_to_hwd(ndarr):
    nd = ndarr.shape[0]
    ndarr = np.split(ndarr, nd, 0) ## split along axis 0
    ndarr = [np.squeeze(n) for n in ndarr] ## make 2D
    ndarr = [np.expand_dims(n,2) for n in ndarr] ## make 3D (h,w,1)
    return np.dstack(ndarr) ## stack 3D (h,w,nd)
#/end activations_to_hwd




def run_net(net, img, rotate=False, layer='conv_classifier'):
    activations = []
    batchsize = net.blobs['data'].shape[0]
    if rotate and batchsize == 1:
        _ = net.forward(data=img_to_caffe(img))
        activ = np.squeeze(net.blobs[layer].data)
        activations.append(activations_to_hwd(activ))

        for rot in range(1,4):
            img_ = np.rot90(img, rot)
            _ = net.forward(data=img_to_caffe(img_))
            activ = np.squeeze(net.blobs[layer].data)
            activ = activations_to_hwd(activ)
            activ = np.rot90(activ, 4-rot)
            activations.append( activ )
        #/end for
        # return np.prod(activations, axis=0)
        return np.mean(activations, axis=0)

    elif rotate and batchsize == 4:
        img_in = [img_to_caffe(np.rot90(img, rot)) for rot in range(4)]
        # img_in = [img_to_caffe(img)]
        # for rot in range(1,4):
        #     img_in.append(img_to_caffe(np.rot90(img, rot)))
        #/end for
        _ = net.forward(data=np.concatenate(img_in, axis=0))  # (4,3,h,w)
        activ = []
        for k in range(4):
            act = np.squeeze(net.blobs[layer].data)
            act = activations_to_hwd(act)
            activ.append(np.rot90(act, 4-k))
        #/end for
        # return np.prod(activations, axis=0)
        return np.mean(activations, axis=0)

    elif not rotate and batchsize == 1:
        _ = net.forward(data=img_to_caffe(img))
        activ = np.squeeze(net.blobs[layer].data)
        return activations_to_hwd(activ)

    else:
        raise ValueError('Rotate was specified, but batchsize was not 1 or 4')
    #/end if
#/end run_net


def process_svs(svs, prob_maps, coordinates, settings):
    # Check inputs
    overlap = settings['overlap']
    scales = settings['scales']
    weights = settings['weights']
    netproto = settings['deploy_proto']
    gpumode = settings['gpumode']
    cnnlayer = settings['cnnlayer']
    n_classes = settings['n_classes']
    prefetch = settings['prefetch']
    do_normalize = settings['do_normalize']
    debug_mode = settings['DEBUGGING']
    rotate = settings['rotate']
    caffe_root = settings['caffe_root']

    svs_info = {}
    svs_info = data_utils.pull_svs_stats(svs, svs_info)
    lvl20_index = svs_info['20x_lvl']
    mult_5x = svs_info['20x_to_5x']


    ## Loop over scales
    pmap_out = []
    for coords, scale, weight in zip(
        coordinates, scales, weights):
        # pmap_scale = np.copy(prob_maps)  ## use this line to enforce default class
        pmap_scale = np.zeros_like(prob_maps)
        # for enforcing the border
        h,w = pmap_scale.shape[:2]
        processed = np.zeros((h,w), dtype=np.bool)

        net = init_net(netproto, weight, caffe_root, gpumode=gpumode)

        print 'Processing {}'.format(scale)
        print 'Using {} tile coordinates'.format(len(coords))

        if scale == '20x':
            mult = 1
            place_mult = 0.25**2
        elif scale == '10x':
            mult = 2
            place_mult = 1/8.
        elif scale == '5x':
            mult = 4
            place_mult = 0.25
        #/end if

        ## A subset for speed
        # indices = np.random.choice(range(len(coords)), 250)
        # coords = [coords[index] for index in indices]
        # print 'Subsetted {} coordinates '.format(len(coords))

        failed_count = 0
        load_size = proc_size = settings['proc_size']
        load_size *= mult
        place_size = proc_size * place_mult
        place_size = int(place_size)

        print 'load_size:', load_size
        print 'place_size:', place_size

        ## Divide the set into n chunks
        if len(coords) < prefetch:
            print 'Coordinates ({}) < prefetch ({})'.format(
                len(coords), prefetch
            )
            coord_split = [coords]
            n_splits = 1
        else:
            n_splits = len(coords) / prefetch
            coord_split = np.array_split(coords, n_splits)
        #/end if

        print 'n splits: ', n_splits

        for nindx, coord_prefetch in enumerate(coord_split):
            ## tile preloading
            preload_start = time.time()
            random.shuffle(coord_prefetch)
            tiles = data_utils.preload_tiles(svs, coord_prefetch,
                    size=(load_size, load_size), level=lvl20_index)
            tiles = [cv2.resize(tile, dsize=(proc_size, proc_size)) for tile in tiles]

            if do_normalize:
                tiles = [cnorm.normalize(tile) for tile in tiles]
            #/end if

            print '{} Tiles preloaded in {:3.3f}s'.format(len(tiles), time.time() - preload_start),

            ## Processing here
            # tiles = [cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY) for tile in tiles]
            cnn_start = time.time()
            tiles = [run_net(net, tile, rotate=rotate, layer=cnnlayer) for tile in tiles]
            print 'CNN finished in {:3.3f}s'.format(time.time() - cnn_start),

            # Resize to fit
            placing_start = time.time()
            tiles = [cv2.resize(tile, dsize=(place_size, place_size)) for tile in tiles]
            coord_prefetch = [(int(x * mult_5x), int(y * mult_5x)) for (x,y) in coord_prefetch]

            # if overlap > 0:
            #     ovp = int(overlap * mult_5x)
            #     place_size_crop = place_size - ovp
            #     bbox = [ovp,
            #             place_size_crop,
            #             ovp,
            #             place_size_crop]
            #     # print 'ovp:', ovp
            #     # print 'bbox:', bbox
            #     # place_size_crop -= ovp
            #     # print 'place_size_crop:', place_size_crop
            #
            #     tiles = [tile[bbox[0]:bbox[1], bbox[2]:bbox[3], :] for tile in tiles]
            #     tiles = [cv2.resize(tile, dsize=(place_size, place_size)) for tile in tiles]
            # #/end if

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
                    # placeholder_out = placeholder[in_out==0]
                    # border = np.mean([tile_out, placeholder_out])
                    placeholder[in_out==0] += tile_out
                    placeholder[in_out==0] /= 2
                    pmap_scale[row:row+place_size, col:col+place_size, :] = placeholder
                else:
                    pmap_scale[row:row+place_size, col:col+place_size, :] = tile
                #/end if
                processed[row:row+place_size, col:col+place_size] = True
                # except:
                #     print 'failed {} <-- {}'.format((row, col), tile.shape)
                #     failed_count += 1
                #/end try
            #/end for tile, (row,col)
            print 'Placing done in {:3.3f}s'.format(time.time() - placing_start)
        #/end for nindx, coord_prefetch

        print 'Failed: {}'.format(failed_count)

        pmap_out.append(pmap_scale)
    #/end for coords, scale, overlap, weight

    return pmap_out
#/end process_svs
