import numpy as np
from tensorflow.keras import backend as K


def dice_coef(y_true: np.ndarray, y_pred: np.ndarray, smooth: int = 1) -> float:
    """
    Compute the dice coefficient between the ground truth and the prediction.
    Parameters
    ----------
    y_true : np.ndarray
        The ground truth mask.
    y_pred : np.ndarray
        The predicted mask.
    smooth : int, optional
        The smoothing factor. The default is 1.

    Returns
    -------
    float
        The dice coefficient.

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the dice loss between the ground truth and the prediction.
    Parameters
    ----------
    y_true : np.ndarray
        The ground truth mask.
    y_pred : np.ndarray
        The predicted mask.

    Returns
    -------
    float
        The dice loss.
    """
    return 1 - dice_coef(y_true, y_pred)

