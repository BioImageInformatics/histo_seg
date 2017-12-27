## Histo-Seg-2
Histo-Seg is a skeleton for rapid prototyping of image analysis methods applied to digital pathology whole slide images. It uses the [openslide](http://openslide.org) library to read from `svs` image pyramids. Really it's a set of functions strung together with a couple "pipeline" scripts. Development is mostly to facilitate certain projects. Histo-Seg-2 (this repo) is a considerably simplified version of v1. These functions provide the same core operations, much quicker, and in ~1/2 or fewer lines of code.


**November, 2017** Moved over to Tensorflow via `tfmodels` (LINK).


![flow_overview]


### Example Use Cases
1. prostate cancer growth patterns, from manual annotation
2. clear cell renal cancer microenvironment, automatic transfer of Immunohistochemistry annotation
3. WSI Image-to-Image translation for H\&E to IHC or IF (in-progress)
4. WSI feature distributions for hot-spot finding


## Workflow
### Installation
A partial list of package dependecies:
* python 2.7
* openslide-python
* numpy
* scipy
* matplotlib
* OpenCV 2
* TensorFlow >= 1.4

### Training
Training procedure moved to `tfmodels` (LINK).

#### Processing
Processing happens in 3 phases:
* Data preparation from Whole Slide Images (WSI) and low-level ROI finding
* High-resolution processing
* Gather and stitch

The script `histoseg.py` will run the WSI pipeline on `svs` files found in a user defined directory.

Example usage:
```
$ python ~/histo-seg/core/histoseg.py --slide=/path/to/slide.svs --settings=/path/to/settings.pkl
```

The example script uses a RAM drive available by default on Ubuntu OS at `/dev/shm`.
An alternative is to mount a RAMDISK to a path of your choosing using `tmpfs`.
Then again, reading from an SSD or fast HDD could be fast enough, in which case set `ramdisk` to `None`.


### Features visualization & WSI feature heat maps
IN PLANNING

### License
Please provide citation if you use this library for your research.

```
BIBTEX
```

Copyright 2017 BioImageInformatics Lab, Cedars Sinai Medical Center

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This package was developed with support from the departments of Surgery and Biomedical Sciences at Cedars Sinai Medical Center, Los Angeles, CA.

[flow_overview]: assets/histoseg_overview.png
