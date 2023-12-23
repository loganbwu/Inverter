import numpy as np

def invert(img):
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    return(bit_depth - img)

# Automatically suggest a black point of image
def auto_black_point(img, ignore=0.001, buffer=0.01):
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    return(np.quantile(img, 0.5*ignore, [0,1]) - 0.5*buffer*bit_depth)

# Automatically suggest a white point of image
def auto_white_point(img, ignore=0.001, buffer=0.01):
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    return(np.quantile(img, 1-0.5*ignore, [0,1]) + 0.5*buffer*bit_depth)

#' ignore: {ignore}*100% of the most extreme pixels will not be included in the histogram min/max. Use this to ignore any noise or extreme hot/dead pixels.
#' buffer: we adjust the quantiles to give a bit of room in case there are any small tails of real pixels. The intent is to use AT LEAST the central {1-buffer}*100% of the bit depth, exactly if `ignore==0`.
def normalise(img, black_point=None, white_point=None):

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

# alpha: controls contrast. alpha>1 is more contrast, alpha<1 is less
def contrast(img, alpha=1):
    if alpha == 1:
        return(img)
    n_channels = img.shape[2]
    bit_depth = np.ma.minimum_fill_value(img.dtype)
    

    for i in range(n_channels):
        img[:,:,i] = np.where(img[:,:,i] < 0.5*bit_depth,
                              0.5*bit_depth * (img[:,:,i] / (0.5*bit_depth))**alpha,
                              bit_depth - 0.5*bit_depth * ((bit_depth-img[:,:,i]) / (0.5*bit_depth))**alpha)
    return(img)