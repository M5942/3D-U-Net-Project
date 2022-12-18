import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf


def dice_coef(y_true: np.ndarray | tf.Tensor, y_pred: np.ndarray | tf.Tensor, smooth: int = 1) -> tf.Tensor:
    """
    Compute the dice coefficient between the ground truth and the prediction.
    Parameters
    ----------
    y_true : np.ndarray | tf.Tensor
        The ground truth mask.
    y_pred : np.ndarray | tf.Tensor
        The predicted mask.
    smooth : int, optional
        The smoothing factor. The default is 1.

    Returns
    -------
    tf.Tensor
        The dice coefficient.

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    # The 0.1 is subtracted to account for inflation of metric value due to to_categorical
    # This problem is described in the report
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) - .1


def dice_loss(y_true: np.ndarray | tf.Tensor, y_pred: np.ndarray | tf.Tensor) -> tf.Tensor:
    """
    Compute the dice loss between the ground truth and the prediction.
    Parameters
    ----------
    y_true : np.ndarray | tf.Tensor
        The ground truth mask.
    y_pred : np.ndarray | tf.Tensor
        The predicted mask.

    Returns
    -------
    tf.Tensor
        The dice loss.
    """
    return 1 - dice_coef(y_true, y_pred)

