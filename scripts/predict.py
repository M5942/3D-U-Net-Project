import numpy as np
import nibabel as nib
from patchify import unpatchify
from tensorflow.keras.models import load_model
from .preprocess import load_nifti, preprocess_image, resize_image, get_patches


def predict(model_path: str, image_data: np.ndarray, batch_size: int = 6) -> np.ndarray:
    """
    Predicts the mask of the input image
    Parameters
    ----------
    model_path : str
        Path to the model
    image_data : np.ndarray
        Input image
    batch_size : int
        Batch size

    Returns
    -------
    np.ndarray
        Predicted mask
    """

    # Load the model
    model = load_model(model_path, compile=False)

    # Predict the mask of the input image
    mask = model.predict(image_data, batch_size=batch_size)

    # Return the predicted mask
    return mask


def predict_from_filepath(model_path: str, image_path: str,
                          patch_size: tuple = (64, 64, 64), batch_size: int = 6) -> np.ndarray:
    """
    Predicts the mask of the input image from the image path
    Parameters
    ----------
    model_path : str
        Path to the model
    image_path : str
        Path to the image
    patch_size : tuple
        Size of the patch
    batch_size : int
        Batch size

    Returns
    -------
    np.ndarray
        Predicted mask
    """

    # Load the image
    image_data = load_nifti(image_path)

    # Preprocess the image
    dims = np.array(image_data.shape)
    new_dims = np.round(dims / patch_size[0]) * patch_size[0]
    image_data = preprocess_image(image_data, new_dims)

    patches = get_patches(image_data, patch_size)

    patches = np.expand_dims(patches, axis=(-1))

    masks = predict(model_path, patches, batch_size=batch_size)

    # Accepts the greater probability
    masks = np.argmax(masks, axis=-1)

    # Reshape the mask to make it suitable for unpatchify
    masks = masks.reshape((tuple((new_dims / patch_size[0]).astype(int)) + patch_size))

    # Unpatch the mask
    mask = unpatchify(masks, tuple(new_dims.astype(int)))

    # Resize the mask to the original dimensions
    mask = resize_image(mask, dims)
    mask[mask != 0] = 1

    # Return the predicted mask
    return mask


def predict_multiple(model_path: str, img_path_lst: list[str]) -> list:
    """
    Takes in a list of image paths and predicts the masks for each image using the given model path

    Parameters
    ----------
    model_path : str
        Path to the model
    img_path_lst : list[str]
        List of image paths

    Returns
    -------
    list
        List of predicted masks

    """

    masks = []

    for img in img_path_lst:
        # Predict the mask
        pred = predict_from_filepath(model_path, img)

        masks.append(pred)

    return masks


def save_as_nifti(image: np.ndarray, path: str):
    """
    Saves the given image as a nifti file
    Parameters
    ----------
    image : np.ndarray
        Image to be saved
    path : str
        Path to save the image
    """

    # Create a nifti image
    nifti_image = nib.Nifti1Image(image, np.eye(4))

    # Save the image
    nib.save(nifti_image, path)






