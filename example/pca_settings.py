import cPickle as pickle
import numpy as np

## Defaults
settings = {
## Stuff for the dataset
    'title':            'segnet-5x-tf',
    'n_classes':        4,
    'class_names':      ['LowGrade', 'HighGrade', 'Benign', 'Stroma'],
    'replace_value':    3,
    'colors':           np.array([[235, 40, 40], [40, 235, 40], [40, 40, 245], [0, 0, 0]]),
    'proc_size':        256,
    'scales':           ['5x'],
    'scale_weights':    [1],
    # 'overlap':          64,
    'overlap':          0.1,
    'ramdisk':          '/dev/shm',
## Caffe root, prototxt and weight files
    # 'caffe_root':       '/home/nathan/caffe-segnet-crf/python',
    # 'weights':          ['/home/nathan/histo-seg/semantic-pca/weights/whole_set_512/batchnorm_segnet_basic_pca_250000.caffemodel',
    #                      '/home/nathan/histo-seg/semantic-pca/weights/whole_set_1024/batchnorm_segnet_basic_pca_250000.caffemodel'],
    # 'deploy_proto':     '/home/nathan/histo-seg/semantic-pca/code/segnet_basic_deploy.prototxt',
    # 'cnnlayer':         'prob',
## Tensorflow code, snapshots
    'tfmodel_root':     '/home/nathan/tfmodels',
    'tf_snapshot':      '/home/nathan/tfmodels/experiments/pca128segnet_full/snapshots/segnet.ckpt-95000',
    'tfmodel_name':     'segnet',
## Pull these from the model settings - unfortunately they must match exactly with the original settings
    'conv_kernels':     [64, 128, 256, 512, 512],
    'deconv_kernels':   [64, 128, 256, 512, 512],
    'k_size':           3,
## Options
    'rotate':           False,
    'bayesian':         False,
    'samples':          16,
    'do_post_processing': False,
    'gpumode':          True,
    'do_normalize':     True,
    'output_filenames': ['probability',
                         'argmax',
                         'argmaxRGB',
                         'overlayMAX',
                         'overlaySMTH',
                         'argmaxSMTH',
                         'tissue'],
    'prefetch':         500,
    'DEBUGGING':        False,
}

updates = {
    'title': 'resnet_10x_tf',
    'scales': ['10x'],
    'overlap': 0.25,
    'output_dir': '/home/nathan/histo-seg/durham/resnet10x',
    'tf_snapshot': '/home/nathan/tfmodels/experiments/pca10Xresnet/snapshots/resnet.ckpt-199000',
    'tfmodel_name':     'resnet',
    'conv_kernels':     [64, 128, 256, 512, 512],
    'deconv_kernels':   [64, 128, 256, 512, 512],
    'k_size':           3,
    'bayesian':         False,
    'samples':          16,
    'prefetch':         250,
    'proc_size':        386,
## settings for resnets
    'resnet_stacks':    5,
    'resnet_kernels':   [64, 64, 64, 128],
## Set true for debug mode
    'DEBUGGING': False,
}

if updates['bayesian']:
    print 'BAYESIAN MODE IS BROKEN PLEASE FIX'
    # updates['output_filenames'] = ['probability', 'argmax', 'argmaxRGB', 'overlayMAX', 'tissue', 'variance']

print 'Updating default settings:'
for key, val in sorted(updates.items()):
    print '\t {} --> {}'.format(key, val)

settings.update(**updates)
filename = 'example/{}.pkl'.format(settings['title'])
with open(filename, 'w') as f:
    pickle.dump(settings, f)
print 'Saving settings to:', filename
