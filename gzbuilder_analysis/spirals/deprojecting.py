import numpy as np
from skimage.transform import rotate, rescale
from skimage.util import crop


def deproject_array(arr, angle=0, ba=1):
    ba = min(ba, 1/ba)
    rotated_image = rotate(arr, angle)
    stretched_image = rescale(
        rotated_image,
        (1/ba, 1),
        mode='constant',
        anti_aliasing=True,
        multichannel=False
    )
    crop_amounts = np.repeat(
        np.subtract(stretched_image.shape, arr.shape),
        2
    ).reshape(2, 2) / 2

    return crop(stretched_image, crop_amounts)


def deproject_arm(arm, angle=0, ba=1):
    """Given an array of xy pairs, an axis ratio, and a rotation angle, rotate
    the points about the origin and scale outwards along the y axis
    """
    p = np.deg2rad(-angle)
    rotation_matrix = np.array(
        ((np.cos(p), -np.sin(p)), (np.sin(p), np.cos(p)))
    )
    rotated_arm = np.dot(rotation_matrix, arm.T)
    stretched_arm = rotated_arm.T * (1, 1/ba)
    return stretched_arm


def reproject_arm(arm, angle=0, ba=1):
    return deproject_arm(
        deproject_arm(arm, 0, 1/ba),
        -angle, 1
    )