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

settings = {
    'n_classes': 4,
    'class_names': ['LowGrade', 'HighGrade', 'Benign', 'Stroma'],
    'proc_size': 256,
    'scales': ['10x', '5x'],
    # 'scales': [512-128, 1024-128],
    'overlaps': [64, 64],
    'output_dir': '/home/nathan/histo-seg/pca-dev',
    'weights': ['/home/nathan/histo-seg/semantic-pca/weights/xval_set_0_512/batchnorm_segnet_basic_pca_20170712.SGD_iter_65000.caffemodel',
                '/home/nathan/histo-seg/semantic-pca/weights/xval_set_0_1024/batchnorm_segnet_basic_pca_20170712.SGD_iter_65000.caffemodel'],
    'deploy_proto': '/home/nathan/histo-seg/semantic-pca/code/segnet_basic_deploy.prototxt',
    'title': 'dev'
}

with open('/home/nathan/histo-seg/v2/example/pca_settings.pkl', 'w') as f:
    pickle.dump(settings, f)
