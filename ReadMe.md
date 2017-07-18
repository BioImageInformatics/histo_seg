## Histo-Seg
Histo-Seg is a framework for rapid prototyping of image analysis methods applied to digital pathology whole slide images. It uses the [openslide](http://openslide.org) library to read from `svs` image pyramids. Really it's a set of functions strung together with a couple "pipeline" scripts. Development is mostly to facilitate certain projects. Please leave reviews, feedback and functional suggestions in the Issues section. I'm consistently aiming to simplify the internal code to promote extendability.

See the publication [HERE].

Segmentation and semantic labelling using the [SegNet](http://mi.eng.cam.ac.uk/projects/segnet/) architecture ([github](https://github.com/alexgkendall/caffe-segnet)). An interesting post-processing layer implementing Conditional Random Fields ([CRFasRNN](https://github.com/torrvision/crfasrnn)) is optional. Note that these two architectures use layers unavailable in `caffe-master`. A merger of the two custom branches can be found ([here](https://github.com/nathanin/caffe-segnet-crf)).

Other methods (e.g. FCN) that take in images and produce label matrices can be quickly substituted as an alternative.

It should be straightforward to take models defined in other libraries with Python front-ends (i.e. Tensorflow) and use them in this framework. I'll get on implementing a quick example.


### Example Use Cases
* Use-case 1: prostate cancer growth patterns, from manual annotation

* Use-case 2: clear cell renal cancer microenvironment, via automatic transfer of Immunohistochemistry annotation

* Use-case 3: lung adenocarcinoma growth patterns, with converted Full-Connected deconvolution layers; trained on whole tile examples.

<<<<<<< HEAD

### Preparing data
Training data must be as image label pairs. Each "data" image should be accompanied by a similarly sized "label" image indicating ground truth examples for the classification. The annotations often indicate discrete biological structures or motifs canonically defined by pathologist consensus.

In histopathology, training data must be curated with the domain knowledge of a trained pathologist. Annotation scarcity is a well documented shortcoming in the field (citations), and represents a significant bottleneck in training data-driven models. Therefore, it's common to use data augmentation pre-processing steps which effectively multiply the area used for training. Some data augmentation implemented here includes:
* Random sub-sampling at variable scales
* Color augmentation in LAB space 
=======
## Workflow
### Installation
A partial list of package dependecies:
* openslide-python
* numpy
* scipy
* matplotlib
* OpenCV 2

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
$ python ~/histo-seg/core/data_pipeline.py 512 10 dataset_01
```

This command sub-samples each image 10 times at 512 px windows. It saves the results into a directory `dataset_01`. In histopathology, training data must be curated with the domain knowledge of a trained pathologist. Annotation scarcity is well documented in the field (citations), and represents a significant bottleneck in training data-driven models. Therefore, it's common to use data augmentation pre-processing steps which effectively multiply the area used for training. Some data augmentation implemented here includes:
* Random sub-sampling, including over-sampling
* Color augmentation in LAB space
>>>>>>> 046be63370a9e04946611aa63f0a12cf39891529
* Image rotation

In the future, instead of saving large datasets of individually preprocessed images, it's likely better to store a single reference dataset and apply runtime augmentation.

#### Processing
After a segmentation model is trained, the most interesting application setting is to whole mount slides or biopsies. These are the smallest unit of tissue that pathologists evaluate for the presence and severity of diseased cells. A major aim of this package is to emulate a pathologist's evaluation.

Processing happens in 3 phases:
* Data preparation from Whole Slide Images (WSI) and low-level ROI finding
* High-resolution discretized processing
* Process results agglomeration and report generation

The script `histo_pipeline.py` will run the WSI pipeline on `svs` files found in a user defined directory.

Example usage:
```
$ python ~/histo-seg/core/histo_pipeline.py
```

These phases are implemented as individual packages, together composing the "core" module. Since each phase depends only on the previous phase being completed, they are executable in isolation for fast idea prototyping. For example, I have 10 slides to process. Phase 1 performs the basic tissue-finding and tiling well. There is no longer much need to repeat phase 1 if I want to try options in phases 2 and 3, so we recycle the output of phase 1. This approach comes at the cost of disk space and I/O time. A fast SSD or using a RAM drive are worth considering.

In case you're using Ubuntu, there is a preconfigured RAM drive available at `/dev/shm` that is by default 1/2 of your system's total RAM. A second option is to mount a RAMDISK to a path of your choosing using `tmpfs`.

It was a side goal to allow execution on a massively parallel environment like an AWS cluster, then to pull the results into a central system for further processing, analysis, and long term storage. This goal has yet to be implemented.

### Unsupervised learning
TODO

### Features visualization & WSI feature heat maps
IN PLANNING

### Affiliations
This package was developed with support from the BioImageInformatics Lab at Cedars Sinai Medical Center, Los Angeles, CA.

Questions & comments to ing[dot]nathany[at]gmail[dot]com
