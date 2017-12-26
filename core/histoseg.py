"""
Single image -- use histoseg_batch instead
tensorflow version:
    November, 2017, TF 1.3.0
"""

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

def init_net(settings, sess, gpumode=True):
    tfmodel_root = settings['tfmodel_root']
    tf_snapshot = settings['tf_snapshot']
    conv_kernels = settings['conv_kernels']
    deconv_kernels = settings['deconv_kernels']
    k_size = settings['k_size']
    x_dims = [settings['proc_size'], settings['proc_size'], 3]
    tfmodel_name = settings['tfmodel_name']

    sys.path.insert(0, tfmodel_root)
    try:
        import tfmodels
        print 'Success loading tfmodels'
    except Exception as e:
        print e.__doc__
        print e.message
        print 'ERROR: Failed to load tfmodels from {} (is TensorFlow installed?)'.format(tfmodel_root)
        raise e
    #/end try

    print 'Starting network'
    network_args = {'sess':sess, 'n_classes':settings['n_classes'],
        'conv_kernels':conv_kernels, 'deconv_kernels':deconv_kernels,
        'k_size':k_size, 'x_dims':x_dims}

    ## TODO fix this whole thing
    if tfmodel_name=='vgg':
        net = tfmodels.VGGInference(**network_args)
    elif tfmodel_name=='segnet':
        net = tfmodels.SegNetInference(**network_args)
    elif tfmodel_name=='resnet':
        ## TODO fix this
        network_args = {'sess':sess, 'n_classes':settings['n_classes'],
            'kernels': settings['resnet_kernels'],
            'stacks': settings['resnet_stacks'],
            'k_size':k_size, 'x_dims':x_dims}
        net = tfmodels.ResNetInference(**network_args)

    # net.print_info()

    try:
        print 'Restoring..'
        net.restore(tf_snapshot)
    except:
        print 'Failed to restore snapshot from {}'.format(tf_snapshot)

    return net


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

    print 'Setting up network'
    net = init_net(settings, sess)

    try:
        svs = data_utils.open_slide(svs_ramdisk)

        # start -- do tiling / preprocessing
        print 'Tiling'
        coordinates, prob_maps, background, detailed_bcg = tile.tile_svs(svs, settings)
        print 'Done tiling'

        # keep going
        print 'Entering process procedure'
        process_start = time.time()
        # prob_maps = process_tf.process_svs(svs, prob_maps, coordinates, sess, settings)
        prob_maps = process_tf.process_svs(svs, prob_maps, coordinates, net, settings)
        print
        print 'Processing {} done in {:3.3f}s'.format(svsbase, time.time() - process_start)
        print

        # done?
        # prob_combo, prediction, prediction_rgb, overlay = reconstruct.reconstruct(prob_maps,
        #     svs, detailed_bcg, settings)

        out_image_dict = reconstruct.reconstruct(prob_maps, svs, detailed_bcg, settings)
        out_image_dict['tissue'] = 1-background

        # data_utils.save_result([prob_combo, prediction, prediction_rgb, overlay, 1-background],
        #     svsbase, settings)
        label_img = data_utils.read_label(svs)
        data_utils.save_result(out_image_dict, label_img, svsbase, settings)

        ## If we make it all the way here, move the slide form '/path/not_processed' to '/path/processed'
        data_utils.move_slide(args.slide)

    except Exception as e:
        print e.__doc__
        print e.message
    finally:
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
