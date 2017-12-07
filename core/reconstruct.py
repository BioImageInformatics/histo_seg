'''
reconstruct.py

not reconstructing from tiles this time
. reconstructing proposals from probability maps

'''
import cv2
import numpy as np
import sys
import os

from scipy import stats

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
#/end post_process


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

def filter_probability(probs, thresh=0.5, layers=[0,1]):
    probs_ = np.copy(probs)
    for k in layers:
        probs_[(probs_[:,:,k] < thresh), k] = 0
    #/end for

    return probs_
#end filter_probability



"""
Take a single channel image and sliding window it to smooth the predictions
"""
def sliding_window_smooth(img, window_size=64, stride=16):
    print 'Smoothing with sliding window, input image:', img.shape
    # print 'img', img.shape
    h_0, w_0 = img.shape[:2]

    ## pad with stride, then crop to even number
    # img = np.pad(img, [(stride/2, stride/2), (stride/2, stride/2)], mode='reflect')
    # print 'padded image', img.shape
    # # print 'pad img', img.shape
    # # print 'h_0', h_0, 'w_0', w_0
    # h, w = img.shape
    #
    # crop_h = h_0 % stride
    # crop_w = w_0 % stride
    # img = img[crop_h/2:h-crop_h/2, crop_w/2:w-crop_w/2]

    print 'working on image', img.shape
    # print 'crop img', img.shape
    h, w = img.shape[:2]
    # print 'h', h, 'w', w

    # out_img = np.zeros([h/stride, w/stride], dtype=np.uint8)
    out_img = np.zeros_like(img)
    # print 'out_img', out_img.shape
    hhs = xrange(0, h-window_size, stride)
    wws = xrange(0, w-window_size, stride)
    for x, hh in enumerate(hhs):
        for y, ww in enumerate(wws):
            img_slice = img[hh:hh+window_size, ww:ww+window_size]
            majority = stats.mode(img_slice, axis=None)[0][0]
            # out_img[x,y] = majority
            out_img[hh:hh+window_size, ww:ww+window_size] = majority
            # print 'hh', hh, 'ww', ww, 'x', x, 'y', 'majority', majority

    # print 'out_img', out_img.shape
    return out_img
    # return cv2.resize(out_img, dsize=(w_0, h_0),
    #     interpolation=cv2.INTER_NEAREST)



"""
Take prob_maps, a list of nd-array floats

return a combined nd-array the same shape as prob_maps[0]
and the argmax mask
and false colored H&E
"""
def reconstruct(prob_maps, svs, background, settings):
    scales = settings['scales']
    scale_weights = settings['scale_weights']
    colors = settings['colors']
    replace_value = settings['replace_value']
    do_post_processing = settings['do_post_processing']

    print 'Reconstructing predictions from {} scales'.format(len(scales))

    if do_post_processing:
        print 'Applying openings'
        prob_maps = [post_process(prob) for prob in prob_maps]

    # Take a weighted average
    # prob_map = np.average(prob_maps, weights=scale_weights, axis=0)
    # prob_map = np.prod(prob_maps, axis=0)
    prob_map = np.mean(prob_maps, axis=0)

    # prob_map = filter_probability(prob_map)

    # prob_map = np.mean(prob_maps, axis=0)

    ## Argmax and cast the background
    if background.dtype is not 'bool':
        background = background.astype(np.bool)
    prob_max = np.argmax(prob_map, axis=2)
    print 'Background image: ', background.shape
    print 'Prob_max image: ', prob_max.shape

    ## Smooth prediction
    prob_smooth = sliding_window_smooth(prob_max)
    print 'Prob_smooth image: ', prob_smooth.shape, np.unique(prob_smooth)

    ## Remove background
    print 'Replacing background '
    prob_max[background] = replace_value
    # prob_smooth[background] = replace_value

    # Overlay an H&E
    overlay_h_and_e = data_utils.read_low_level(svs)
    overlay_max, prob_max_color = impose_overlay(prob_max, overlay_h_and_e, colors)
    overlay_smooth, prob_smooth_color = impose_overlay(prob_smooth, overlay_h_and_e, colors)

    return_images = {''}
    return_images = {'probability': prob_map,
                     'argmax': prob_max,
                     'argmaxRGB': prob_max_color,
                     'overlayMAX': overlay_max,
                     'overlaySMTH': overlay_smooth,
                     'argmaxSMTH': prob_smooth_color,
                     }
    return return_images

    # return prob_map, prob_max, prob_max_color, overlay_max, overlay_smooth, prob_smooth_color



def reconstruct_variance(var_maps, background, settings):
    scales = settings['scales']
    scale_weights = settings['scale_weights']
    colors = settings['colors']
    replace_value = settings['replace_value']
    do_post_processing = settings['do_post_processing']

    print 'Reconstructing variances from {} scales'.format(len(scales))

    print 'reconstruct var_maps', len(var_maps)
    #print 'reconstruct var_maps', var_maps.shape, var_maps.dtype, var_maps.min(), var_maps.max()
    var_map = np.mean(var_maps, axis=0)
    print 'reconstruct var_map', var_map.shape, var_map.dtype, var_map.min(), var_map.max()

    # prob_map = filter_probability(prob_map)

    # prob_map = np.mean(prob_maps, axis=0)

    var_total = np.sum(var_map, axis=2)
    print 'reconstruct var_total', var_total.shape, var_total.dtype, var_total.min(), var_total.max()
    var_total = cv2.convertScaleAbs(var_total*255)
    print 'reconstruct var_total', var_total.shape, var_total.dtype, var_total.min(), var_total.max()
    # if background.dtype is not 'bool':
    #     background = background.astype(np.bool)
    # var_total[background] = 0

    # Overlay an H&E
    # overlay = data_utils.read_low_level(svs)
    # overlay, prob_max_color = impose_overlay(prob_max, overlay, colors)

    return var_total
