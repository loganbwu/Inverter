import numpy as np

def invert(img):
    """
    Invert a negative image.
    """
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    return(bit_depth - img)

def auto_black_point(img, ignore=0.001, buffer=0.01):
    """
    Automatically suggest a black point of image based on the histogram.
    :param array-like img: A 3-dimensional array.
    :param float ignore: Discard this edge proportion of the histogram to avoid errors such as hot or dead pixels from skewing the histogram max/min.
    :param float buffer: Adjust the quantiles to give a bit of room due to the `ignore` param. If there is a small tail of valid pixels, they can be included in the float, but extreme values should be clipped. If `ignore == 0`, the normalised image will use exactly (1-buffer) of the bit depth.
    """
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    return(np.quantile(img, 0.5*ignore, [0,1]) - 0.5*buffer*bit_depth)

def auto_white_point(img, ignore=0.001, buffer=0.01):
    """
    Automatically suggest a white point. See documentation for `auto_black_point` for details.
    """
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    return(np.quantile(img, 1-0.5*ignore, [0,1]) + 0.5*buffer*bit_depth)

def normalise(img, black_point=None, white_point=None):
    """
    Normalise an image array between the black and white point.
    :param array-like img: A 3-dimensional array.
    :param array-like black_point: Minimum channel values of the untransformed image, mapped to 0. Pixel values smaller than this are clipped.
    :param array-like white_point: Maximum channel values of the untransformed image, mapped to the maximum data type value (bit depth). Pixel values larger than this are clipped.
    """

    if black_point is None:
        black_point = auto_black_point(img)
    if white_point is None:
        white_point = auto_white_point(img)

    n_channels = img.shape[2]
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    
    for i in range(n_channels):
        img[:,:,i] = np.clip((img[:,:,i] - black_point[i]) /
                              (white_point[i] - black_point[i]) * bit_depth,
                              0, bit_depth)
    return(img)

def contrast(img, alpha=1):
    """
    Adjust contrast using a sigmoid curve.
    :param array-like img: A 3-dimensional array.
    :param float alpha: Curve steepness. >1 increases contrast, and <1 decreases contrast.
    """
    if alpha == 1:
        return(img)
    n_channels = img.shape[2]
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    

    for i in range(n_channels):
        img[:,:,i] = np.where(img[:,:,i] < 0.5*bit_depth,
                              0.5*bit_depth * (img[:,:,i] / (0.5*bit_depth))**alpha,
                              bit_depth - 0.5*bit_depth * ((bit_depth-img[:,:,i]) / (0.5*bit_depth))**alpha)
    return(img)