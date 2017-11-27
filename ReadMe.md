## Histo-Seg-2
Histo-Seg is a skeleton for rapid prototyping of image analysis methods applied to digital pathology whole slide images. It uses the [openslide](http://openslide.org) library to read from `svs` image pyramids. Really it's a set of functions strung together with a couple "pipeline" scripts. Development is mostly to facilitate certain projects. Histo-Seg-2 (this repo) is a considerably simplified version of v1. These functions provide the same core operations, much quicker, and in ~1/2 or fewer lines of code.

This repo was used for the SPIE Medical Imaging, Digital Pathology paper:
```
bibtex
```

Segmentation and semantic labelling using the [SegNet](http://mi.eng.cam.ac.uk/projects/segnet/) architecture ([github](https://github.com/alexgkendall/caffe-segnet)). An interesting post-processing layer implementing Conditional Random Fields ([CRFasRNN](https://github.com/torrvision/crfasrnn)) is optional. Note that these two architectures use layers unavailable in `caffe-master`. A merger of the two custom branches can be found ([here](https://github.com/nathanin/caffe-segnet-crf)).

Other methods (e.g. FCN) that take in images and produce label matrices can be quickly substituted as an alternative.

**November, 2017** Moved over to Tensorflow via `tfmodels` (LINK).


### Example Use Cases
1. prostate cancer growth patterns, from manual annotation
2. clear cell renal cancer microenvironment, automatic transfer of Immunohistochemistry annotation
3. lung adenocarcinoma growth patterns, with converted Full-Connected deconvolution layers; trained on whole tile examples.
4. WSI Image-to-Image translation for H\&E to IHC or IF (in-progress)
5. WSI feature distributions for hot-spot finding


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
* High-resolution discretized processing
* Process results agglomeration and report generation

The script `histoseg.py` will run the WSI pipeline on `svs` files found in a user defined directory.

Example usage:
```
$ python ~/histo-seg/core/histoseg.py --slide=/path/to/slide.svs --settings=/path/to/settings.pkl
```

The example script uses a RAM drive available at `/dev/shm` that is by default 1/2 of your system's total RAM. A second option is to mount a RAMDISK to a path of your choosing using `tmpfs`.


### Features visualization & WSI feature heat maps
IN PLANNING

### License
Please provide citation if you use this library for your research.

Copyright 2017 Nathan Ing

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
This package was developed with support from the BioImageInformatics Lab at Cedars Sinai Medical Center, Los Angeles, CA.
