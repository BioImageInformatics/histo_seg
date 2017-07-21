'''
Color transfer of a reference LAB vector mean & standard deviation.

E. Reinhard "Color transfer between images" <em>IEEE Comput. Graph. Appl.</em> vol. 21 no. 5 pp. 34-41 Sep./Oct. 2001.

image
    3-channel RGB space image
target
    3x2 numpy ndarray float64
verbose
    boolean choice to print feedback to stdout
'''


# from scipy import misc
# import skimage as ski
import numpy as np
import cv2


def normalize(image, target=None, verbose=False):
    if target is None:
        target = np.array([[148.60, 41.56], [169.30, 9.01], [105.97, 6.67]])
        # target = np.array( [[68.33, 23.10], [19.09, 16.44], [-11.33, 8.97]] )
    # if verbose:
    #     print "Color normalization:"
    #     # print "Recieved {} image".format(image.dtype),
    #     print "Recieved {} image".format(type(image)),


    M = image.shape[0]
    N = image.shape[1]

    # if verbose:
    #     print " size {}rows, {}cols".format(str(M), str(N))

    whitemask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    whitemask = whitemask > 215

    # if verbose:
    #     print "Taking it into LAB space"
    imagelab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    imageL, imageA, imageB = cv2.split(imagelab)

    # mask is valid when true
    imageLM = np.ma.MaskedArray(imageL, whitemask)
    imageAM = np.ma.MaskedArray(imageA, whitemask)
    imageBM = np.ma.MaskedArray(imageB, whitemask)

    ## Sometimes STD is near 0, or 0; add epsilon to avoid div by 0
    epsilon = 1e-11

    imageLMean = imageLM.mean()
    imageLSTD = imageLM.std() + epsilon

    imageAMean = imageAM.mean()
    imageASTD = imageAM.std() + epsilon

    imageBMean = imageBM.mean()
    imageBSTD = imageBM.std() + epsilon

    if verbose:
        print "Mean/Std vector: [[{:3.3f},{:3.3f}], [{:3.3f},{:3.3f}], [{:3.3f},{:3.3f}]]".format(
            imageLMean, imageLSTD, imageAMean, imageASTD, imageBMean, imageBSTD
        )
        # print "Lmean: {}, Lstd: {}".format(str(imageLMean), str(imageLSTD))
        # print "Amean: {}, Astd: {}".format(str(imageAMean), str(imageASTD))
        # print "Bmean: {}, Bstd: {}".format(str(imageBMean), str(imageBSTD))

    # normalization in lab
    imageL = (imageL - imageLMean) / imageLSTD * target[0][1] + target[0][0]
    imageA = (imageA - imageAMean) / imageASTD * target[1][1] + target[1][0]
    imageB = (imageB - imageBMean) / imageBSTD * target[2][1] + target[2][0]

    imagelab = cv2.merge((imageL, imageA, imageB))
    imagelab = np.clip(imagelab, 0, 255)
    imagelab = imagelab.astype(np.uint8)

    # Back to RGB space
    returnimage = cv2.cvtColor(imagelab, cv2.COLOR_LAB2RGB)
    returnimage[whitemask] = image[whitemask]

    return returnimage
