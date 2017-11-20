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


tensorflow version:
    November, 2017, TF 1.3.0
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
import process_tf
import reconstruct
import data_utils


try:
    import tensorflow as tf
except:
    print 'ERROR loading tensorflow'



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

    for key, value in sorted(settings.items()):
        print '\t {}: {}'.format(key, value)

    ## Initialize the output file by recording the settings
    # for key in settings.iterkeys():
    #     print '{}: {}'.format(key, settings[key])

    svsbase = data_utils.svs_name(args.slide)
    svs_ramdisk = data_utils.transfer_to_ramdisk(args.slide, ramdisk)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    try:
        svs = data_utils.open_slide(svs_ramdisk)

        # start -- do tiling / preprocessing
        print 'Tiling'
        coordinates, prob_maps, background, detailed_bcg = tile.tile_svs(svs, settings)
        print 'Done tiling'

        # keep going
        print 'Entering process procedure'
        process_start = time.time()
        prob_maps = process_tf.process_svs(svs, prob_maps, coordinates, sess, settings)
        print
        print 'Processing {} done in {:3.3f}s'.format(svsbase, time.time() - process_start)
        print

        # done?
        prob_combo, prediction, prediction_rgb, overlay = reconstruct.reconstruct(prob_maps,
            svs, detailed_bcg, settings)
        data_utils.save_result([prob_combo, prediction, prediction_rgb, overlay, 1-background],
            svsbase, settings)
    except Exception as e:
        print e.__doc__
        print e.message

    print 'Removing {}'.format(svs_ramdisk)
    data_utils.delete_from_ramdisk(svs_ramdisk)

    print 'Closing session'
    sess.close()
#/end main


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--slide')
    p.add_argument('--settings')
    p.add_argument('--output_dir')

    args = p.parse_args()

    main(args)
    # test(args)
