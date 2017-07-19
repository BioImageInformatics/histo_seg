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
    filled-in output probability images

r():
input:
    probability images
    processing params
output:
    smoothed output
    proposal label image

new things:
    - to save space and make for cleaner code, pickle the settings and use those
'''

import argparse
import cv2
import sys
import cPickle as pickle

sys.path.insert(0, '.')
import tile
import process
import reconstruct
import data_utils


def main(args):
    # check arguments
    assert os.path.exists(args.settings)
    assert os.path.exists(args.slide)
    with open(args.settings, 'r') as f:
        settings = pickle.load(f)
    #/end with

    svs = data_utils.open_slide(args.slide)

    # start -- do tiling / preprocessing
    coordinates, prob_images = tile.tile_svs(svs, settings)

    # keep going
    prob_images = process.process_svs(prob_images, coordinates, settings)

    # done?
    outputs = reconstruct.reconstruct_svs(prob_images, settings)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--slide')
    p.add_argument('--settings')

    args = p.parse_args()

    main(args)
