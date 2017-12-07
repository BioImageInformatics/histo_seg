import cPickle as pickle
import numpy as np

settings = {
## Stuff for the dataset
    'title':            'resnet-tf',
    'n_classes':        4,
    'class_names':      ['LowGrade', 'HighGrade', 'Benign', 'Stroma'],
    'replace_value':    3,
    'colors':           np.array([[235, 40, 40], [40, 235, 40], [40, 40, 245], [0, 0, 0]]),
    'proc_size':        256,
    'scales':           ['10x'],
    'scale_weights':    [1],
    'overlap':          0,
## Stuff for the loading and saving
    'output_dir':       '/home/nathan/histo-seg/semantic-pca/tensorflow/resnet10x',
    'ramdisk':          '/dev/shm',
## Caffe root, prototxt and weight files
    # 'caffe_root':       '/home/nathan/caffe-segnet-crf/python',
    # 'weights':          ['/home/nathan/histo-seg/semantic-pca/weights/whole_set_512/batchnorm_segnet_basic_pca_250000.caffemodel',
    #                      '/home/nathan/histo-seg/semantic-pca/weights/whole_set_1024/batchnorm_segnet_basic_pca_250000.caffemodel'],
    # 'deploy_proto':     '/home/nathan/histo-seg/semantic-pca/code/segnet_basic_deploy.prototxt',
    # 'cnnlayer':         'prob',
## Tensorflow code, snapshots
    'tfmodel_root':     '/home/nathan/tfmodels',
    'tf_snapshot':      '/home/nathan/tfmodels/experiments/pca256resnet/snapshots/resnet.ckpt-30000',
    #'tfmodel_root':     '/Users/nathaning/_projects/tfmodels',
    #'tf_snapshot':      '/Users/nathaning/_projects/tfmodels/experiments/pca128resnet/snapshots/resnet.ckpt-50000',
    'tfmodel_name':     'resnet',
    # 'tf_snapshot':      '/home/nathan/tfmodels/experiments/pca128/snapshots/vgg_segmentation.ckpt-148250',
## Pull these from the model settings - unfortunately they must match exactly with the original settings
    'conv_kernels':     [64, 128, 256],
    'deconv_kernels':   [64, 128, 256],
    'k_size':           3,
## Options
    'rotate':           False,
    'bayesian':         False,
    'samples':          32,
    'do_post_processing': False,
    'gpumode':          True,
    'do_normalize':     True,
    'output_filenames': ['probability', 'argmax', 'argmaxRGB', 'overlay', 'tissue'],
    #'output_filenames': ['probability', 'argmax', 'argmaxRGB', 'overlay', 'tissue', 'variance'],
    'prefetch':         500,
    'DEBUGGING':        False,
}

filename = 'example/resnet_tfmodels_10x.pkl'
with open(filename, 'w') as f:
    pickle.dump(settings, f)
print filename
