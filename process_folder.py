import numpy as np
from exif import Image
import cv2
import os
from PIL import Image as PillowImage
from PIL import ExifTags
from libtiff import TIFFimage
import tifffile
# from pylibtiff import TIFFimage

from transformations import normalise, contrast, auto_black_point, auto_white_point, invert


folder_in = '/Users/wu.l/Downloads/3187'
to_invert = False
folder_in = '/Users/wu.l/Desktop/2023-12-23'
to_invert = True

edit_tag = '-Edit'
files_in = sorted(os.listdir(folder_in))
files_in = [x for x in files_in if not edit_tag in x and any([x.lower().endswith(y) for y in ['tif', 'tiff', 'jpg', 'jpeg']])]

if files_in:
    for i, f in enumerate(files_in):
        print(f'[{i+1}/{len(files_in)}] Processing {f}')
        
        f_in = os.path.join(folder_in, f)
        basename, ext = os.path.splitext(f_in)
        f_out = os.path.join(basename + edit_tag + ext)

        img = PillowImage.open(f_in)
        img_format = img.format
        if img_format == 'JPEG':
            img_exif = img.getexif()
            img = np.array(img)
            # img_exif = {k: v for k, v in img_exif.items() if k in ExifTags.TAGS.keys()}
            # img_exif.pop(33723, None)
        if img_format == 'TIFF':
            img = cv2.imread(f_in, cv2.IMREAD_UNCHANGED)

        img = invert(img) if to_invert else img
        black_point = auto_black_point(img)
        white_point = auto_white_point(img)
        img = normalise(img, black_point, white_point)
        img = contrast(img, 1)

        if img_format == 'JPEG':
            # PIL preserves metadata but needs libtiff installed for TIFFs
            img = PillowImage.fromarray(img)
            img.save(f_out, exif=img_exif, optimize=True, quality=95)
        if img_format == 'TIFF':
            cv2.imwrite(f_out, img, params=[cv2.IMWRITE_TIFF_COMPRESSION, 8])
