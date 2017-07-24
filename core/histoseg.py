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

module_dir, module_name = os.path.split(__file__)
sys.path.insert(0, module_dir)
import tile
import process
import reconstruct
import data_utils


def test(args):
    print 'Got slide: ', args.slide
    print 'Got settings: ', args.settings
#/end test

def main(args):
    # check arguments
    assert args.slide and args.settings
    assert os.path.exists(args.settings)
    assert os.path.exists(args.slide)
    with open(args.settings, 'r') as f:
        settings = pickle.load(f)
    #/end with
    if args.output_dir:
        settings['output_dir'] = args.output_dir
    #/end if
    ramdisk = settings['ramdisk']

    ## Initialize the output file by recording the settings
    # for key in settings.iterkeys():
    #     print '{}: {}'.format(key, settings[key])

    svsbase = data_utils.svs_name(args.slide)
    svs_ramdisk = data_utils.transfer_to_ramdisk(args.slide, ramdisk)

    svs = data_utils.open_slide(svs_ramdisk)

    # start -- do tiling / preprocessing
    coordinates, prob_maps, background = tile.tile_svs(svs, settings)

    # keep going
    process_start = time.time()
    prob_maps = process.process_svs(svs, prob_maps, coordinates, settings)
    print
    print 'Processing {} done in {:3.3f}s'.format(svsbase, time.time() - process_start)
    print

    # done?
    prob_combo, prediction, prediction_rgb, overlay = reconstruct.reconstruct(prob_maps,
        svs, background, settings)
    data_utils.save_result([prob_combo, prediction, prediction_rgb, overlay],
        svsbase, settings)

    print 'Removing {}'.format(svs_ramdisk)
    data_utils.delete_from_ramdisk(svs_ramdisk)
#/end main


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--slide')
    p.add_argument('--settings')
    p.add_argument('--output_dir')

    args = p.parse_args()

    main(args)
    # test(args)
