import sys
from openslide import OpenSlide
import cPickle as pickle

svs = OpenSlide('/dev/shm/1305400.svs')
with open('/home/nathan/histo-seg/v2/example/pca_settings.pkl', 'r') as f:
    settings = pickle.load(f)


sys.path.insert(0, '/home/nathan/histo-seg/v2/core')
import tile
import data_utils

coordinates, prob_maps = tile.tile_svs(svs, settings)

tile.DEBUGGING_pull_tiles_from_coords(svs, coordinates[1], '/dev/shm/tmp', level=-1)
