from PIL import Image as PillowImage
import numpy as np
import cv2

def read_img(path: str):
    if any([path.lower().endswith(x) for x in ['jpg', 'jpeg']]):
        img = PillowImage.open(path)
        img_exif = img.getexif()
        img = np.array(img)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_exif = None

    return(img, img.dtype, img_exif)

def write_img(path: str, img, img_exif=None):
    if any([path.lower().endswith(x) for x in ['jpg', 'jpeg']]):
        # PIL preserves metadata but needs libtiff installed for TIFFs
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = PillowImage.fromarray(img)
        img.save(path, exif=img_exif, optimize=True, quality=95)
    else:
        if img.dtype not in [np.uint8, np.uint16]:
            img = img.astype(np.uint16)
        cv2.imwrite(path, img, params=[cv2.IMWRITE_TIFF_COMPRESSION, 8]) # 8=adobe deflate
