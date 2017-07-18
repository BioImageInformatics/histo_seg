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

'''

import argparse
import cv2
import sys

sys.path.insert(0, '.')
import tile
import process
import reconstruct
import data_utils

def parse_args(args):
    pass


def main(**kwargs):
    pass


if __name__ == '__main__':
    args = parse_args(sys.argv)
