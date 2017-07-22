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
    'title':            'dev',
    'n_classes':        4,
    'class_names':      ['LowGrade', 'HighGrade', 'Benign', 'Stroma'],
    'replace_value':    3,
    'colors':           np.array([[235, 40, 40], [40, 235, 40], [40, 40, 245], [0, 0, 0]]),
    'proc_size':        256,
    'scales':           ['10x', '5x'],
    'scale_weights':    [1,1],
    'scale_indices':    [-2, -1],
    'overlaps':         [16, 16],
## Stuff for the loading and saving
    'output_dir':       '/home/nathan/histo-seg/semantic-pca/analysis_wsi',
    'ramdisk':          '/dev/shm',
## Caffe prototxt and weight files
    'weights':          ['/home/nathan/histo-seg/semantic-pca/weights/xval_set_0_512/batchnorm_segnet_basic_pca_20170712.SGD_iter_65000.caffemodel',
                         '/home/nathan/histo-seg/semantic-pca/weights/xval_set_0_1024/batchnorm_segnet_basic_pca_20170712.SGD_iter_65000.caffemodel'],
    'deploy_proto':     '/home/nathan/histo-seg/semantic-pca/code/segnet_basic_deploy.prototxt',
## Options
    'rotate':           True,
    'do_post_processing': False,
    'gpumode':          True,
    'cnnlayer': 'prob',
    'do_normalize':     True,
    'output_filenames': ['probability', 'argmaxRGB', 'argmax', 'overlay'],
    'prefetch':         1000,
}

with open('example/pca_settings.pkl', 'w') as f:
    pickle.dump(settings, f)
