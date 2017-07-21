## Histo-Seg-2
Histo-Seg is a framework for rapid prototyping of image analysis methods applied to digital pathology whole slide images. It uses the [openslide](http://openslide.org) library to read from `svs` image pyramids. Really it's a set of functions strung together with a couple "pipeline" scripts. Development is mostly to facilitate certain projects. Histo-Seg-2 (this repo) is a considerably simplified version of v1. These functions provide the same core operations, much quicker, and in ~1/2 or fewer lines of code.

This repo was used for the conference proceeding [SPIE].

Segmentation and semantic labelling using the [SegNet](http://mi.eng.cam.ac.uk/projects/segnet/) architecture ([github](https://github.com/alexgkendall/caffe-segnet)). An interesting post-processing layer implementing Conditional Random Fields ([CRFasRNN](https://github.com/torrvision/crfasrnn)) is optional. Note that these two architectures use layers unavailable in `caffe-master`. A merger of the two custom branches can be found ([here](https://github.com/nathanin/caffe-segnet-crf)).

Other methods (e.g. FCN) that take in images and produce label matrices can be quickly substituted as an alternative.

It should be straightforward to take models defined in other libraries with Python front-ends (i.e. Tensorflow) and use them in this framework. In reality, it's just a very specific data prefetching tool.


### Example Use Cases
* Use-case 1: prostate cancer growth patterns, from manual annotation

* Use-case 2: clear cell renal cancer microenvironment, via automatic transfer of Immunohistochemistry annotation

* Use-case 3: lung adenocarcinoma growth patterns, with converted Full-Connected deconvolution layers; trained on whole tile examples.

## Workflow
### Installation
A partial list of package dependecies:
* python 2.7
* openslide-python
* numpy
* scipy
* matplotlib
* OpenCV 2
* Caffe, Caffe2, Tensorflow, Torch, etc.

### Training
#### Preparing data
Training data follows the "data - label" pair model. Each "data" image should be accompanied by a similarly sized "label" image indicating ground truth examples for the classification. The annotations often indicate discrete biological structures or motifs canonically defined by pathologist consensus. Each "data - label" pair ought to have the same root file name. E.g.:

```
Training/
  feature/
    img001.jpg
    img002.jpg
  label/
    img001.png
    img002.png
```

Once the images are prepared you can use the `data_pipeline.py` script to multiply the raw data into training and validation sets.

```
$ python ~/histo-seg2/scripts/data_pipeline.py 512 10 dataset_01
```

This command sub-samples each image 10 times at 512 px windows. It saves the results into a directory `dataset_01`. In histopathology, training data must be curated with the domain knowledge of a trained pathologist. Annotation scarcity is well documented in the field (citations), and represents a significant bottleneck in training data-driven models. Therefore, it's common to use data augmentation pre-processing steps which effectively multiply the area used for training. Some data augmentation implemented here includes:
* Random sub-sampling, including over-sampling
* Color augmentation in LAB space
* Image rotation

In the future, instead of saving large datasets of individually preprocessed images, it's likely better to store a single reference dataset and apply runtime augmentation.

#### Processing
After a segmentation model is trained, the most interesting application setting is to whole mount slides or biopsies. These are the smallest unit of tissue that pathologists evaluate for the presence and severity of diseased cells. A major aim of this package is to emulate a pathologist's evaluation.

Processing happens in 3 phases:
* Data preparation from Whole Slide Images (WSI) and low-level ROI finding
* High-resolution discretized processing
* Process results agglomeration and report generation

The script `histoseg.py` will run the WSI pipeline on `svs` files found in a user defined directory.

Example usage:
```
$ python ~/histo-seg/core/histoseg.py --slide=/path/to/slide.svs --settings=/path/to/settings.pkl
```

In case you're using Ubuntu, there is a preconfigured RAM drive available at `/dev/shm` that is by default 1/2 of your system's total RAM. A second option is to mount a RAMDISK to a path of your choosing using `tmpfs`. If requested, the slide can be temporarily stored in RAM during processing.

It was a side goal to allow execution on a massively parallel environment like an AWS cluster, then to pull the results into a central system for further processing, analysis, and long term storage. Alternatively, to use the fact that data prefetching and augmentation takes far longer than inference, we could use the `multiprocessing` library to implement a data Queueing system. (https://github.com/BVLC/caffe/issues/3607). This goal has yet to be implemented.

### Unsupervised learning
TODO

### Semi-supervised learning
One application for this pipeline is to semi-automate data collection. One could process a section of slide, then have a domain expert evaluate the annotation. This process would continue ad-nauseam, until the model stops making errors, or the pathologist finds something better to do.

### Adversarial learning
To produce stronger heat maps globally, we could train an adversary net to distinguish between CNN proposal segmentation, and segmentation performed by a human pathologist. Very interesting.

### Features visualization & WSI feature heat maps
IN PLANNING

### Affiliations
This package was developed with support from the BioImageInformatics Lab at Cedars Sinai Medical Center, Los Angeles, CA.

Questions & comments to ing[dot]nathany[at]gmail[dot]com
