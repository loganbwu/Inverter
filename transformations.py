import numpy as np

def invert(img):
    """
    Invert a negative image.
    """
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    return(bit_depth - img)

def auto_black_point(img, clip=0.001):
    """
    Automatically suggest a black point of image based on the histogram.
    :param array-like img: A 3-dimensional array.
    :param float clip: Discard this edge proportion of the histogram to avoid errors such as hot or dead pixels from skewing the histogram max/min.
    """
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    return(np.quantile(img, 0.5*clip, [0,1]))

def auto_white_point(img, clip=0.001):
    """
    Automatically suggest a white point. See documentation for `auto_black_point` for details.
    """
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    return(np.quantile(img, 1-0.5*clip, [0,1]))

def normalise(img, black_point=None, white_point=None, buffer=0.01):
    """
    Normalise an image array between the black and white point.
    :param array-like img: A 3-dimensional array.
    :param array-like black_point: Minimum channel values of the untransformed image, mapped to 0. Pixel values smaller than this are clipped.
    :param array-like white_point: Maximum channel values of the untransformed image, mapped to the maximum data type value (bit depth). Pixel values larger than this are clipped.
    :param float buffer: Adjust the quantiles to give a bit of room due to the `ignore` param. If there is a small tail of valid pixels, they can be included in the float, but extreme values should be clipped. If `clip == 0`, the normalised image will use exactly (1-buffer) of the bit depth.
    """

    """
    f(x) = mx + b
    f(0) = buffer/2
    f(1) = 1 - buffer/2
    b = buffer/2
    m + b = 1 - buffer/2
    m + buffer/2 = 1 - buffer/2
    m = 1 - buffer/2 - buffer/2
    m = 1-buffer
    """

    if black_point is None:
        black_point = auto_black_point(img)
    if white_point is None:
        white_point = auto_white_point(img)

    n_channels = img.shape[2]
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    
    for i in range(n_channels):
        # Original steps
        # from_0_to_1 = (img[:,:,i] - black_point[i]) / (white_point[i] - black_point[i])
        # buffered = from_0_to_1 * (1-buffer) + buffer/2
        # scaled = buffered * bit_depth
        # img[:,:,i] = np.clip(scaled, 0, bit_depth)

        img[:,:,i] = np.clip(((img[:,:,i] - black_point[i]) / (white_point[i] - black_point[i]) * (1-buffer) + buffer/2) * bit_depth, 0, bit_depth)
    return(img)

def adjust_contrast(img, alpha=1):
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

def adjust_gamma(img, gamma):
    if gamma == 1:
        return(img)
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    img[:,:,:] = ((img[:,:,:] / bit_depth) ** gamma ) * bit_depth
    return(img)