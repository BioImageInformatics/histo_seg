'''
tensorflow version:
    November, 2017, TF 1.4.0
    - (requires tf.nn.selu and tf.contrib.nn.alpha_dropout)
'''

import argparse
import cv2, sys, os, glob
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

'''
Set up tensorflow session, model, load snapshot
'''
def init_net(settings, sess, gpumode=True):
    tfmodel_root = settings['tfmodel_root']
    tf_snapshot = settings['tf_snapshot']
    conv_kernels = settings['conv_kernels']
    deconv_kernels = settings['deconv_kernels']
    k_size = settings['k_size']
    x_dims = [settings['proc_size'], settings['proc_size'], 3]

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
    net = tfmodels.SegNetInference(sess=sess,
        n_classes=settings['n_classes'],
        conv_kernels=conv_kernels,
        deconv_kernels=deconv_kernels,
        k_size=k_size,
        x_dims=x_dims)
    net.print_info()

    try:
        print 'Restoring..'
        net.restore(tf_snapshot)
    except:
        print 'Failed to restore snapshot from {}'.format(tf_snapshot)

    return net

#/end init_net


def main(args):
    # check arguments
    assert args.source_dir and args.settings
    assert os.path.exists(args.settings)
    assert os.path.exists(args.source_dir)
    with open(args.settings, 'r') as f:
        settings = pickle.load(f)
    #/end with
    if args.output_dir:
        settings['output_dir'] = args.output_dir
    #/end if
    ramdisk = settings['ramdisk']

    slide_list = sorted(glob.glob(os.path.join(
        args.source_dir, '*.svs' )))
    assert len(slide_list) >= 1

    for key, value in sorted(settings.items()):
        print '\t {}: {}'.format(key, value)

    ## Initialize the output file by recording the settings
    # for key in settings.iterkeys():
    #     print '{}: {}'.format(key, settings[key])
    print 'Initializing TensorFlow Session'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    ## set up TF
    print 'Setting up network'
    net = init_net(settings, sess)

    for slide in slide_list:
        svsbase = data_utils.svs_name(slide)
        svs_ramdisk = data_utils.transfer_to_ramdisk(slide, ramdisk)

        try:
            svs = data_utils.open_slide(svs_ramdisk)

            # start -- do tiling / preprocessing
            print 'Tiling'
            coordinates, prob_maps, background, detailed_bcg = tile.tile_svs(svs, settings)
            print 'Done tiling'

            # keep going
            print 'Processing {}'.format(svsbase)
            process_start = time.time()
            prob_maps = process_tf.process_svs(svs, prob_maps, coordinates, net, settings)
            print
            print '..done in {:3.3f}s'.format(time.time() - process_start)
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
    #/end for

    print 'Closing session'
    sess.close()
#/end main


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source_dir')
    p.add_argument('--settings')
    p.add_argument('--output_dir')

    args = p.parse_args()

    main(args)
    # test(args)
