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
    'title':            'vgg-tf',
    'n_classes':        4,
    'class_names':      ['LowGrade', 'HighGrade', 'Benign', 'Stroma'],
    'replace_value':    3,
    'colors':           np.array([[235, 40, 40], [40, 235, 40], [40, 40, 245], [0, 0, 0]]),
    'proc_size':        256,
    'scales':           ['5x'],
    'scale_weights':    [1],
    'overlap':          64,
## Stuff for the loading and saving
    'output_dir':       '/home/nathan/histo-seg/semantic-pca/tensorflow/vgg',
    'ramdisk':          '/dev/shm',
## Caffe root, prototxt and weight files
    'caffe_root':       '/home/nathan/caffe-segnet-crf/python',
    'weights':          ['/home/nathan/histo-seg/semantic-pca/weights/whole_set_512/batchnorm_segnet_basic_pca_250000.caffemodel',
                         '/home/nathan/histo-seg/semantic-pca/weights/whole_set_1024/batchnorm_segnet_basic_pca_250000.caffemodel'],
    'deploy_proto':     '/home/nathan/histo-seg/semantic-pca/code/segnet_basic_deploy.prototxt',
## Tensorflow code, snapshots
    'tfmodel_root':     '/home/nathan',
    # 'tf_snapshot':      '/home/nathan/tfmodels/experiments/pca256/snapshots/vgg_segmentation.ckpt-20750',
    'tf_snapshot':      '/home/nathan/tfmodels/experiments/pca128/snapshots/vgg_segmentation.ckpt-148250',
## Options
    'rotate':           False,
    'do_post_processing': False,
    'gpumode':          True,
    'cnnlayer':         'prob',
    'do_normalize':     True,
    'output_filenames': ['probability', 'argmax', 'argmaxRGB', 'overlay', 'tissue'],
    'prefetch':         1000,
    'DEBUGGING':        False,
}

### Assertions to check settings validity
# assert len(settings['scales']) == len(settings['weights'])
# assert len(settings['scales']) == len(settings['overlaps'])
# assert n_classes == len(settings['class_names'])
# assert n_classes == settings['colors'].shape[0]

filename = 'example/vgg_tensorflow_settings.pkl'
with open(filename, 'w') as f:
    pickle.dump(settings, f)
print filename
