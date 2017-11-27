'''
pickle settings

scales:
512, 1024

overlap:
64

output_dir:
/home/nathan/histo-seg/pca-dev
'''

import cPickle as pickle
import numpy as np

settings = {
## Stuff for the dataset
    'title':            'segnet-tf',
    'n_classes':        4,
    'class_names':      ['LowGrade', 'HighGrade', 'Benign', 'Stroma'],
    'replace_value':    3,
    'colors':           np.array([[235, 40, 40], [40, 235, 40], [40, 40, 245], [0, 0, 0]]),
    'proc_size':        256,
    'scales':           ['5x'],
    'scale_weights':    [1],
    'overlap':          64,
## Stuff for the loading and saving
    'output_dir':       '/home/nathan/histo-seg/semantic-pca/tensorflow/segnet_bayes',
    'ramdisk':          '/dev/shm',
## Caffe root, prototxt and weight files
    'caffe_root':       '/home/nathan/caffe-segnet-crf/python',
    'weights':          ['/home/nathan/histo-seg/semantic-pca/weights/whole_set_512/batchnorm_segnet_basic_pca_250000.caffemodel',
                         '/home/nathan/histo-seg/semantic-pca/weights/whole_set_1024/batchnorm_segnet_basic_pca_250000.caffemodel'],
    'deploy_proto':     '/home/nathan/histo-seg/semantic-pca/code/segnet_basic_deploy.prototxt',
## Tensorflow code, snapshots
    'tfmodel_root':     '/home/nathan/tfmodels',
    'tf_snapshot':      '/home/nathan/tfmodels/experiments/pca128segnet/snapshots/segnet.ckpt-21000',
    # 'tf_snapshot':      '/home/nathan/tfmodels/experiments/pca128/snapshots/vgg_segmentation.ckpt-148250',
## Pull these from the model settings - unfortunately they must match exactly with the original settings
    'conv_kernels':     [64, 64, 128, 256],
    'deconv_kernels':   [64, 64, 128],
    'k_size':           5,
## Options
    'rotate':           False,
    'bayesian':         True,
    'samples':          25,
    'do_post_processing': False,
    'gpumode':          True,
    'cnnlayer':         'prob',
    'do_normalize':     False,
    'output_filenames': ['probability', 'argmax', 'argmaxRGB', 'overlay', 'tissue'],
    'prefetch':         1000,
    'DEBUGGING':        False,
}

filename = 'example/segnet_tfmodels_5x_bayes.pkl'
with open(filename, 'w') as f:
    pickle.dump(settings, f)
print filename
