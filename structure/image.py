""" Simple module to handle modifying images """
from typing import Tuple
from pathlib import Path
import numpy as np
import imutils
import cv2

DEPTH_GRAY = 1  # the depth of grayscale image
DEPTH_COLOR = 3  # the depth of a color image
DEPTH_COLOR_ALPHA = 4  # the depth of a color and alpha image

# convert a color from a byte depth to the next depth
# key = (from, to), value = the opencv conversion constant
DEPTH_CONVERSIONS = {
    (1, 3): cv2.COLOR_GRAY2RGB,
    (3, 1): cv2.COLOR_RGB2GRAY,
    (1, 4): cv2.COLOR_GRAY2RGBA,
    (3, 4): cv2.COLOR_RGB2RGBA,
    (4, 3): cv2.COLOR_RGBA2RGB,
    (4, 1): cv2.COLOR_RGBA2GRAY
}

# a representation of the opencv interpolation constants
INTERPOLATIONS = {
    'area': cv2.INTER_AREA,
    'nearest': cv2.INTER_NEAREST,
    'lancz': cv2.INTER_LANCZOS4
}


def has_depth(img: np.ndarray) -> bool:
    """ Determines if an image has a depth element """
    return len(img.shape) >= 3


def image_dims(img: np.ndarray) -> Tuple[int, int]:
    """ Gets the dimensions of an image in wxh format """
    if (len(img.shape) == 3 and img.shape[0] < img.shape[-1]) or len(img.shape) == 4:
        return img.shape[2], img.shape[1]
    elif 2 <=len(img.shape) < 4:
        return img.shape[1], img.shape[0]
    else:
        raise RuntimeError('can not get the dimensions of a flat or timed array')


def fix_depth(img: np.ndarray, depth: int=DEPTH_COLOR):
    """ Fixes the image depth if it doesn't match the current depth in bytes
    
    :param img: the input image
    :param depth: the image depth in bytes
    """
    if len(img.shape) <= 1:  # the array has been flattened
        raise RuntimeError('we cannot assume the dimesions of a flattened or empty array') 
    elif len(img.shape) == 2:  # we only have wxh and a grayscale image
        if depth == DEPTH_GRAY:
            img = img.reshape(img.shape + (depth,))
        else:
            img = cv2.cvtColor(img, DEPTH_CONVERSIONS[(DEPTH_GRAY, depth)])
    elif img.shape[-1] != depth:  # this is a multi-page/time image
        if DEPTH_GRAY <= img.shape[-1] < DEPTH_COLOR:  # convert from grayscale to another depth
            if depth == DEPTH_GRAY:
                img = img[:, :, 0].reshape(img.shape[:2] + (DEPTH_GRAY,))
            else:
                img = cv2.cvtColor(img[:, :, 0], DEPTH_CONVERSIONS[(DEPTH_GRAY, depth)])
        elif img.shape[-1] == DEPTH_COLOR and depth != DEPTH_COLOR:
            img = cv2.cvtColor(img, DEPTH_CONVERSIONS[(DEPTH_COLOR, depth)])
            if depth == DEPTH_GRAY:  # opencv will return just a wxh image when in reality we need a wxhxd image
                img = img.reshape(img.shape + (DEPTH_GRAY,))
        else:
            raise RuntimeError('we cannot fix the depth of a multi-page/time array of shape {}'.format(img.shape))
    return img


def fix_simple_depth(img: np.ndarray) -> np.ndarray:
    """ Fixes the simple grayscale depth of the images """
    if not has_depth(img):
        img = img.reshape(img.shape + (DEPTH_GRAY,))
    return img


def resize_aspect(img: np.ndarray, width: int=None, height: int=None, interp=cv2.INTER_AREA) -> np.ndarray:
    """ Resizes an image keeping its aspect ratio

    :param img: the image to resize
    :param width: the width of the new image
    :param height: the height of the new image
    :param interp: the interpolation of the image
    :return: a resized image with the new aspect ratio
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = img.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return img

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(img, dim, interpolation=interp)

    # return the resized image
    return resized


def resize_image(img: np.ndarray, dims: Tuple[int, int], interp: str='nearest', aspect=None, fix_depth=True) -> np.ndarray:
    """ Resizes an image to the specified dimensions

    :param img: the image to resize
    :param dims: the dimensions of the image wxh
    :param interp: the interpolation of the image (default: 'nearest')
    :param aspect: keep the aspect ratio (default: None)
    :param fix_depth: fix the image to have a depth if it doesn't have it (default: True)
    :return: the resized image
    """
    if dims[0] != img.shape[1] or dims[1] != img.shape[0]:
        try:
            if (None in dims or aspect == 'aspect') and aspect != 'none':  # we want to keep some aspect ratio
                img = imutils.resize(img, dims[0], dims[1], INTERPOLATIONS[interp])
            elif aspect == 'contain':  # we want to contain the image inside the dims bounding box
                width, height = dims[:2]
                if img.shape[0] >= img.shape[1]:  # adjust by height
                    width = None
                else:
                    height = None

                # get the initial image with the fixed depth
                cp_img = fix_simple_depth(resize_aspect(img, width, height, INTERPOLATIONS[interp]))

                # paste the resized image into the container if the dimensions don't match
                if dims[0] != img.shape[1] or dims[1] != img.shape[0]:
                    cpy = np.zeros((dims[1], dims[0], img.shape[-1] if has_depth(img) else 1), img.dtype)
                    cpy[0:cp_img.shape[0], 0:cp_img.shape[1]] = cp_img  # place the resized image into the black background image
                    cp_img = cpy
                
                # replace the original image with the contained one
                img = cp_img
            else:
                img = cv2.resize(img, dims[:2], INTERPOLATIONS[interp])  # distortion is allowed
        except KeyError:
            raise ValueError('invalid interpolation method {}. Please use one of the following {}'.format(interp, list(INTERPOLATIONS.keys())))
        
        if not has_depth(img) and fix_depth:  # fix the grayscale depth if applicable
            img = img.reshape(img.shape + (DEPTH_GRAY,))
    return img


def resize_stack(stack: np.ndarray, dims: Tuple[int, int], interp: str='nearest', aspect=None, fix_depth=True) -> np.ndarray:
    """ Resizes a stack of images to the specified dimensions

    :param img: the image stack to resize
    :param dims: the dimensions of the image wxh
    :param interp: the interpolation of the image (default: 'nearest')
    :param aspect: keep the aspect ratio (default: None)
    :param fix_depth: fix the image to have a depth if it doesn't have it (default: True)
    :return: the resized image stack
    """
    if len(stack.shape) != 4:
        raise RuntimeError('invalid image stack shape of {}'.format(stack.shape))

    # make sure the sizes are correct if not we have to make a copy of the array and loop through to resize the stack
    if dims[0] != stack.shape[2] or dims[1] != stack.shape[1]:
        cp_stack = np.zeros((stack.shape[0],) + (dims[1], dims[0]) + (stack.shape[-1],), stack.dtype)  # this will assume the same dtype and depth as the current stack
        
        # loop through each layer of the stack
        for i in range(stack.shape[0]):
            cp_stack[i] = resize_image(stack[i], dims, interp=interp, aspect=aspect, fix_depth=True)

        # replace the current stack
        stack = cp_stack
    return stack


def resize_image_nparray(img: np.ndarray, dims: Tuple[int, int], interp: str='nearest', aspect=None, fix_depth=True):
    if img is None:
        return img
    elif (len(img.shape) == 3 and img.shape[0] < img.shape[-1]) or len(img.shape) == 4:  # most likely this image is stacked if we have fewer elements in our first shape then the last
        is_stack = True  # we can assume the array has probably already been fixed
    else:
        is_stack = False  # let's assume this array is just a single image
    
    ref_fix_shape = False  # flag to determine if we need to fix the last depth shape in case of grayscale images
    if is_stack:
        if len(img.shape) == 3:
            img = img.reshape(img.shape + (1,))
            ref_fix_shape = True
        img = resize_stack(img, dims, interp, aspect, fix_depth)
        if ref_fix_shape:
            return img.reshape(img.shape[:3])
        return img

    if len(img.shape) == 2:
        img = img.reshape(img.shape + (1,))
        ref_fix_shape = True
    img = resize_image(img, dims, interp, aspect, fix_depth)
    if ref_fix_shape:
        return img.reshape(img.shape[:2])
    return img