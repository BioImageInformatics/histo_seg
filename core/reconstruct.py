'''
reconstruct.py

not reconstructing from tiles this time
. reconstructing proposals from probability maps

'''
import cv2
import numpy as np
import sys
import os

module_dir, module_name = os.path.split(__file__)
sys.path.insert(0, module_dir)
import data_utils

'''
Take an (h,w,n) shaped image,
apply some openings to each n
'''
def post_process(prob):
    n = prob.shape[2]
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    layers = np.split(prob, n, 2)
    layers = [ cv2.morphologyEx(layer, cv2.MORPH_OPEN, kernel_cross) for layer in layers ]
    layers = [ cv2.morphologyEx(layer, cv2.MORPH_OPEN, kernel_circle) for layer in layers ]

    return np.dstack(layers)


'''
Take
arg1: label image
arg2: h&e image in rgb space

return false colored rgb
'''
def impose_overlay(label, target, colors):
    r = np.zeros_like(label)
    g = np.zeros_like(label)
    b = np.zeros_like(label)

    assert len(np.unique(label)) <= colors.shape[0]

    for index in np.unique(label):
        print 'Replacing label {} with {}'.format(index, colors[index, :])
        r[label==index] = colors[index, 0]
        g[label==index] = colors[index, 1]
        b[label==index] = colors[index, 2]
    #/end for

    # replace label with color coded label
    label = np.dstack([r,g,b])
    output = target * 0.6 + label * 0.4

    # return output[:,:,::-1], label
    return output, label
#/end impose_overlay


# def impose_overlay(label, target):
#     if label.shape[-1] > 3:
#         label = label[:,:,:3]
#     #/end if
#
#     label = label[:,:,::-1]
#     output = target*0.6 + label*0.4
#
#     output = cv2.convertScaleAbs(output)
#     # return output[:,:,::-1], label
#     return output
#
# #/end impose_overlay

def filter_probability(probs, thresh=0.5, layers=[0,1,2]):
    probs_ = np.copy(probs)
    for k in layers:
        probs_[(probs_[:,:,k] < thresh), k] = 0
    #/end for

    return probs_
#end filter_probability

'''
Take prob_maps, a list of nd-array floats

return a combined nd-array the same shape as prob_maps[0]
and the argmax mask
and false colored H&E

'''
def reconstruct(prob_maps, svs, background, settings):
    scales = settings['scales']
    scale_weights = settings['scale_weights']
    colors = settings['colors']
    replace_value = settings['replace_value']
    do_post_processing = settings['do_post_processing']

    print 'Reconstructing from {} scales'.format(len(scales))

    if do_post_processing:
        print 'Applying openings'
        prob_maps = [post_process(prob) for prob in prob_maps]

    # Take a weighted average
    # prob_map = np.average(prob_maps, weights=scale_weights, axis=0)
    prob_map = np.prod(prob_maps, axis=0)

    prob_map = filter_probability(prob_map)

    # prob_map = np.mean(prob_maps, axis=0)

    # Argmax
    if background.dtype is not 'bool':
        background = background.astype(np.bool)
    prob_max = np.argmax(prob_map, axis=2)
    prob_max[background] = replace_value

    # Overlay an H&E
    overlay = data_utils.read_low_level(svs)
    overlay, prob_max_color = impose_overlay(prob_max, overlay, colors)

    return prob_map, prob_max, prob_max_color, overlay
