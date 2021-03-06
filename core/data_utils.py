"""

data_utils.py

utility functions for read/write

"""
from openslide import OpenSlide
import cv2
import numpy as np
import time
import sys
import os
import shutil

module_dir, module_name = os.path.split(__file__)
sys.path.insert(0, module_dir)
import colorNormalization as cnorm


def move_slide(slide, srcdir='not_processed', destdir='processed'):
    # slide_base = os.path.basename(slide)
    dest_slide = slide.replace(srcdir, destdir)
    print 'Moving {} to {}'.format(slide, dest_slide)
    shutil.move(slide, dest_slide)


def open_slide(slide):
    return OpenSlide(slide)
#/end retrn_slide

def pull_svs_stats(svs, svs_info):
    app_mag = svs.properties['aperio.AppMag']
    level_dims = svs.level_dimensions

    # Find 20X level:
    if app_mag == '20':  # scanned @ 20X
        print 'Working 20X slide'
        svs_info['app_mag'] = '20'
        svs_info['20x_lvl'] = 0
        svs_info['20x_dim'] = level_dims[0][::-1]
        svs_info['20x_downsample'] = svs.level_downsamples[0]
        svs_info['mult_5x'] = 1/16.
    elif app_mag == '40':  # scanned @ 40X
        # print 'WARNING CHECK 40X NUMBERS'
        print 'Working 40X slide'
        svs_info['app_mag'] = '40'
        svs_info['20x_lvl'] = 1
        svs_info['20x_dim'] = level_dims[1][::-1]
        svs_info['20x_downsample'] = svs.level_downsamples[1]
        svs_info['mult_5x'] = 1/64.
    #/end if
    svs_info['20x_to_5x'] = 1/16.
    svs_info['lvl0_dim'] = level_dims[0]
    svs_info['low_downsample'] = svs.level_downsamples[-1]

    return svs_info
#/end pull_svs_stats

def read_region(svs, x, y, level, size, flip_channels=False, verbose=False):
    # Utility function because openslide loads as RGBA
    if verbose:
        print 'Reading SVS: ({},{}), LEVEL {}, SIZE={}'.format(
            x,y,level,size)
    #/end if

    ## TODO Check if region is out of range for the requested level
    # level_dims = svs.level_dimensions[level]
    # assert x > 0 and y > 0, print 'data_utils.read_region: X and Y must be positive'
    # ## Not sure of the order
    # assert x + size[0] < level_dims[1]
    # assert y + size[1] < level_dims[0]

    img = svs.read_region((x,y), level, size)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if flip_channels:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ## This actually needs to be here

    return img


""" Load up a bunch of tiles as an ndarray """
def preload_tiles(svs, coords, level, size, as_ndarray=False, normalize=False):
    tiles = [read_region(svs, x, y, level, size= size)
             for (y,x) in coords]

    # For completeness. Moved normalize outside & after resize
    if normalize:
        tiles = [cnorm.normalize(tile) for tile in tiles]

    # tiles = [cv2.resize(tile, dsize=(size,size)) for tile in tiles]

    if as_ndarray:
        tiles = [np.expand_dims(tile, axis=0) for tile in tiles]
        tiles = np.concatenate(tiles, axis=0)

    return tiles


""" Read a label image from svs file """
def read_label(svs):
    print 'Reading label image'
    img = svs.associated_images['label']
    img = np.asarray(img)
    img = np.rot90(img, 3)
    print 'Returning label image size {}'.format(img.shape)

    return img


""" Just a helper """
def read_low_level(svs, verbose=False):
    if svs.level_count == 4 and svs.properties['aperio.AppMag'] == '20':
        low_index = svs.level_count - 2
    else:
        low_index = svs.level_count - 1
    #/end if

    img = read_region(svs, 0, 0, low_index,
        svs.level_dimensions[low_index], flip_channels=True, verbose=verbose)
    return img


""" copy file to ramdisk """
def transfer_to_ramdisk(filename, destination):
    newname = os.path.basename(filename)
    newname = os.path.join(destination, newname)

    try:
        shutil.copyfile(filename, newname)
        print 'Copied {} to {}'.format(filename, newname)
    except:
        print 'Failed to transfer {} to {}'.format(filename, newname)
        return 0

    return newname

""" delete a file """
def delete_from_ramdisk(filename):
    try:
        os.remove(filename)
    except:
        return 0

"""
Return the basename
"""
def svs_name(pathname):
    base = os.path.basename(pathname)
    a, ext = os.path.splitext(base)
    return a


"""
utility to save results
"""
def save_result(imgs, label_img, svsbase, settings):
    output_dir = settings['output_dir']
    output_filenames = settings['output_filenames']
    n_classes = settings['n_classes']

    # assert len(imgs) == len(output_filenames)

    # 'output_filenames': ['probability',
    #                      'argmax',
    #                      'argmaxRGB',
    #                      'overlayMAX',
    #                      'overlaySMTH',
    #                      'argmaxSMTH',
    #                      'tissue'],
    ## Save the label image

    filename_ = os.path.join(output_dir, svsbase+'_label.jpg')
    print 'Saving {}'.format(filename_)
    cv2.imwrite(filename_, label_img)

    for filename in output_filenames:
        try:
            img = imgs[filename]
            if 'argmax' in filename:
                filename_ = os.path.join(output_dir, svsbase+'_'+filename+'.png')
                print 'Saving {}'.format(filename_)
                cv2.imwrite(filename_, img)
            elif filename == 'probability':
                ## Uncomment to save full-res 3D stack
                # print 'Writing probability npy {}'.format(img.shape)
                # filename_ = os.path.join(output_dir, svsbase+'_'+filename+'.npy')
                # np.save(filename_, img)
                print 'Writing probability jpg {}'.format(img.shape)
                filename_ = os.path.join(output_dir, svsbase+'_'+filename+'.jpg')
                cv2.imwrite(filename_, img*255/img.max())
            elif filename == 'tissue':
                ext = '.png'
                mult = 255
                filename_ = os.path.join(output_dir, svsbase+'_'+filename+ext)
                print 'Saving {}'.format(filename_)
                cv2.imwrite(filename_, img*mult)
            elif 'overlay' in filename:
                ext = '.jpg'
                filename_ = os.path.join(output_dir, svsbase+'_'+filename+ext)
                print 'Saving {}'.format(filename_)
                cv2.imwrite(filename_, img)
            elif filename == 'variance':
                ext = '.jpg'
                filename_ = os.path.join(output_dir, svsbase+'_'+filename+ext)
                print 'Saving {}'.format(filename_)
                cv2.imwrite(filename_, img)
            else:
                print 'Filename {} does not match a mode. Edit in settings'
                continue

        except Exception as e:
            print e.message
            print e.__doc__
            print 'Error in save_result'
            continue
