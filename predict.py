import numpy as np
import nibabel as nib
from patchify import unpatchify
from tensorflow.keras.models import load_model
from preprocess import load_nifti, preprocess_image, resize_image, get_patches


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

    # TODO: Maybe make model parameter instead of model_path. Both? Try to load as model and if it fails,
    #  load as weights

    # Load the model
    model = load_model(model_path)

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

    # TODO: Load model then and get input shape from model?

    # Preprocess the image
    dims = np.array(image_data.shape)
    new_dims = np.round(dims / patch_size[0]) * patch_size[0]
    image_data = preprocess_image(image_data, new_dims)

    patches = get_patches(image_data, patch_size)

    patches = np.expand_dims(patches, axis=(-1))

    masks = predict(model_path, patches, batch_size=batch_size)

    # Accepts values equal to or greater than 0.5 as mask and the rest as background
    masks = np.argmax(masks, axis=-1)

    masks = masks.reshape((tuple((new_dims / patch_size[0]).astype(int)) + patch_size))

    # Unpatch the mask
    mask = unpatchify(masks, tuple(new_dims.astype(int)))

    # Resize the mask
    mask = resize_image(mask, dims)
    mask[mask != 0] = 1
    mask = mask.astype(np.uint8)

    # Return the predicted mask
    return mask


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


def get_IoU(gt, pred):
    """
    Calculates the Intersection over Union (IoU) of the predicted mask and the ground truth mask
    Parameters
    ----------
    gt : np.ndarray
        Ground truth mask
    pred : np.ndarray
        Predicted mask

    Returns
    -------
    float
        Intersection over Union
    """

    from tensorflow.keras.metrics import BinaryIoU

    iou = BinaryIoU()

    iou_val = iou(gt, pred).numpy()

    # Return the IoU
    return iou_val


if __name__ == '__main__':
    import os

    model_path = r"models\cerebellum_model.h5"
    image_path = r"data\cerebellum_data\image\tc1_276388-ob_c.nii.gz"

    # Predict the mask
    mask = predict_from_filepath(model_path, image_path)

    prediction_path = os.path.basename(image_path).split(".")[0] + "_prediction.nii"

    # Save the mask
    save_as_nifti(mask, prediction_path)
