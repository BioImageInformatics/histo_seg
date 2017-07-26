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
    'output_dir':       '/home/nathan/histo-seg/semantic-pca/analysis_wsi/segnet_basic',
    'ramdisk':          '/dev/shm',
## Caffe root, prototxt and weight files
    # 'caffe_root':       '/Users/nathaning/software/caffe-segnet-crf/python',
    'caffe_root':       '/home/nathan/caffe-segnet-crf/python',
    'weights':          ['/home/nathan/histo-seg/semantic-pca/weights/xval_set_0_512/batchnorm_segnet_basic_pca_20170712.SGD_iter_65000.caffemodel',
                         '/home/nathan/histo-seg/semantic-pca/weights/xval_set_0_1024/batchnorm_segnet_basic_pca_20170712.SGD_iter_65000.caffemodel'],
    'deploy_proto':     '/home/nathan/histo-seg/semantic-pca/code/segnet_basic_deploy.prototxt',
## Options
    'rotate':           True,
    'do_post_processing': False,
    'gpumode':          True,
    'cnnlayer':         'prob',
    'do_normalize':     True,
    'output_filenames': ['probability', 'argmax', 'argmaxRGB', 'overlay'],
    'prefetch':         1000,
    'DEBUGGING':        False,
}

### Assertions to check settings validity
# assert len(settings['scales']) == len(settings['weights'])
# assert len(settings['scales']) == len(settings['overlaps'])
# assert n_classes == len(settings['class_names'])
# assert n_classes == settings['colors'].shape[0]


with open('example/pca_settings.pkl', 'w') as f:
    pickle.dump(settings, f)
