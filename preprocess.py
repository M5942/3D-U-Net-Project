from typing import Iterable
import patchify
import nibabel as nib
from skimage.transform import resize
import numpy as np


def resize_image(image: np.ndarray, image_size: Iterable[int]) -> np.ndarray:
    """
    Resizes an image to a given size.

    Parameters
    ----------
    image : np.ndarray
        The image to resize.
    image_size : Iterable[int]
        The size of the image.

    Returns
    -------
    np.ndarray
        The resized image.
    """

    return resize(image, image_size, anti_aliasing=True)


def get_patches(image: np.ndarray, patch_size: tuple[int, int, int] = (64, 64, 64), stride: int = 64) -> np.ndarray:
    """
    Extracts patches from an image.
    Parameters
    ----------
    image : np.ndarray
        The image to extract patches from.
    patch_size : tuple[int, int, int], optional
        The size of the patches to extract, by default (64, 64, 64)
    stride : int, optional
        The stride of the patches, by default 64. If the stride is less than the patch size, the patches will overlap.

    Returns
    -------
    np.ndarray
        The extracted patches.
    """

    # Get the patches
    patches = patchify.patchify(image, patch_size, step=stride)
    # Get the shape of the patches
    shape = patches.shape
    # Reshape the patches
    patches = patches.reshape((shape[0] * shape[1] * shape[2], shape[3], shape[4], shape[5]))

    # Return the patches
    return patches


def load_nifti(path: str) -> np.ndarray:
    """
    Loads a nifti image.
    Parameters
    ----------
    path : str
        The path to the nifti image.
    Returns
    -------
    np.ndarray
        The loaded image.
    """

    # Load the image
    image = nib.load(path).get_fdata()

    # Return the image data
    return image


def preprocess_image(image: np.ndarray, new_dims: np.ndarray) -> np.ndarray:

    """
    Preprocesses an image.
    Parameters
    ----------
    image : np.ndarray
        The image to preprocess.
    new_dims : np.ndarray
        The new dimensions of the image.
    Returns
    -------
    np.ndarray
        The preprocessed image.
    """

    # Resize the image
    image = resize_image(image, new_dims)

    # Preprocess the image
    image /= 255

    return image


def preprocess_mask(mask: np.ndarray, new_dims: np.ndarray) -> np.ndarray:

    """
    Preprocesses a mask.
    Parameters
    ----------
    mask : np.ndarray
        The mask to preprocess.
    new_dims : np.ndarray
        The new dimensions of the mask.
    Returns
    -------
    np.ndarray
        The preprocessed mask.
    """

    # Resize the mask
    mask = resize_image(mask, new_dims)

    # Set the mask values to 0 or 1
    # mask[mask != 0] = 1

    # Return the preprocessed mask
    return mask
