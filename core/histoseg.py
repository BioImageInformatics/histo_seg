'''
New version histoseg.py

the old one was too big and complicated and all-round could be way better

Data flows like this:

x.svs --t(x)--> tiles --p(x)--> process --r(x)--> output.png

t():
input:
    svs
    tile params
output:
    coordinate list
    blank output images

p():
input:
    svs
    coordinate list
    tile params
    blank output images
output:
    filled-in output prob_maps

r():
input:
    prob_maps
    processing params
output:
    smoothed output
    proposal label image

new things:
    - to save space in the code, pickle the settings and use those
'''

import argparse
import cv2
import sys
import os
import cPickle as pickle
import time

sys.path.insert(0, '.')
import tile
import process
import reconstruct
import data_utils


def main(args):
    # check arguments
    #/end if

    assert args.slide and args.settings
    assert os.path.exists(args.settings)
    assert os.path.exists(args.slide)
    with open(args.settings, 'r') as f:
        settings = pickle.load(f)
    #/end with
    if args.output_dir:
        settings['output_dir'] = args.output_dir
    #/end if

    svsbase = data_utils.svs_name(args.slide)
    svs = data_utils.open_slide(args.slide)

    # start -- do tiling / preprocessing
    coordinates, prob_maps, background = tile.tile_svs(svs, settings)

    # keep going
    process_start = time.time()
    prob_maps = process.process_svs(svs, prob_maps, coordinates, settings)
    print
    print 'Processing done in {:3.3f}s'.format(time.time() - process_start)
    print

    # done?
    prob_combo, prediction, overlay = reconstruct.reconstruct(prob_maps,
        svs, background, settings)
    data_utils.save_result([prob_combo, prediction, overlay],
        svsbase, settings)

    # print 'Code home in', os.path.realpath(__file__)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--slide')
    p.add_argument('--settings')
    p.add_argument('--output_dir')

    args = p.parse_args()

    main(args)
