import numpy as np
from exif import Image
import cv2
import os
from PIL import Image as PillowImage
from PIL import ExifTags
from libtiff import TIFFimage
import tifffile
# from pylibtiff import TIFFimage

from transformations import *
from imageio import read_img, write_img
valid_extensions = ('tif', 'tiff', 'jpg', 'jpeg')


folder_in = '/Users/wu.l/Downloads/1145/12294'
to_invert = False
# folder_in = '/Users/wu.l/Desktop/Scan test'
# to_invert = True

edit_tag = '-Edit'
files_in = sorted(os.listdir(folder_in))
files_in = [x for x in files_in if not edit_tag in x and x.lower().endswith(valid_extensions)]

folder_out = os.path.join(folder_in, 'Edits')

if files_in:
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    for i, f in enumerate(files_in):
        print(f'[{i+1}/{len(files_in)}] Processing {f}')
        
        f_in = os.path.join(folder_in, f)
        basename, ext = os.path.splitext(f)
        f_out = os.path.join(folder_out, basename + edit_tag + ext)

        img, native_dtype, img_exif = read_img(f_in)

        img = invert(img, native_dtype) if to_invert else img
        black_point = auto_black_point(img)
        white_point = auto_white_point(img)
        img = normalise(img, native_dtype, black_point, white_point)
        img = adjust_contrast(img, native_dtype, 1)
        img = adjust_gamma(img, native_dtype, 1)

        write_img(f_out, img, img_exif)
