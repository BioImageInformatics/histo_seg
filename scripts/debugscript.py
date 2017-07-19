import sys
from openslide import OpenSlide
import cPickle as pickle

settingsfile = '/Users/nathaning/_projects/histo-seg-v2/example/pca_settings.pkl'
core_dir = '/Users/nathaning/_projects/histo-seg-v2/core'
debug_dir = '/Users/nathaning/_projects/histo-seg-v2/data'

svs = OpenSlide('/Users/nathaning/Dropbox/projects/semantic_pca/data/1305462.svs')
with open(settingsfile, 'r') as f:
    settings = pickle.load(f)

sys.path.insert(0, core_dir)
import tile
import data_utils

coordinates, prob_maps, _ = tile.tile_svs(svs, settings)

tile.DEBUGGING_pull_tiles_from_coords(svs, coordinates[1], debug_dir, level=-1)
tile.DEBUGGING_pull_tiles_from_coords(svs, coordinates[0], debug_dir, level=-2)
