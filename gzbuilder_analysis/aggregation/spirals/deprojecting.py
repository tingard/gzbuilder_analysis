import numpy as np
from skimage.transform import rotate, rescale
from skimage.util import crop


def deproject_array(arr, angle=0, ba=1):
    """Use scikit-image to deproject an image given an ellipticity and position
    angle
    """
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
    """Given an array of xy pairs, an axis ratio, and a rotation angle
    (in degrees), rotate the points about the origin and scale outwards along
    the y axis
    """
    # a = -np.deg2rad(angle)
    rotation_matrix = np.array(
        ((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle)))
    )
    rotated_arm = np.dot(rotation_matrix, arm.T)
    stretched_arm = rotated_arm.T * (1/ba, 1)
    return stretched_arm


def reproject_arm(arm, angle=0, ba=1):
    """Undo the deprojection of a spiral arm
    """
    return deproject_arm(
        deproject_arm(arm, 0, 1/ba),
        -angle, 1
    )


def change_wcs(points, wcs_in, wcs_out):
    """Convert an poly-line from one WCS to another
    """
    return wcs_out.all_world2pix(
        wcs_in.all_pix2world(points, 0),
        0
    )
