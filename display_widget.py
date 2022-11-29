import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive, fixed, IntSlider, Dropdown
from IPython.display import display


def show_prediction(axes: str, slice_num: int, image: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = .5):
    """
    Show the predicted mask and ground truth over the image.
    Parameters
    ----------
    axes : str
        The axes being displayed.
    slice_num : int
        The slice number being displayed.
    image : np.ndarray
        The image being displayed.
    gt : np.ndarray
        The ground truth mask.
    pred : np.ndarray
        The predicted mask.
    alpha : float, optional
    The opacity of the masks.

    Returns
    -------

    """

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    images = None

    # Gets the correct slice from the correct axis
    if axes == "xy":
        images = (image[:, :, slice_num], gt[:, :, slice_num], pred[:, :, slice_num])
    if axes == "yz":
        images = (image[slice_num, :, :], gt[slice_num, :, :], pred[slice_num, :, :])
    if axes == "xz":
        images = (image[:, slice_num, :], gt[:, slice_num, :], pred[:, slice_num, :])

    # Display the image and ground truth if the ground truth is given
    # The image is displayed by default
    ax1.imshow(images[0], cmap="gray")
    if gt is not None:
        ax1.imshow(images[1], cmap="gray", alpha=alpha)
        ax1.set_title("Ground Truth")
        ax1.axis("off")

    # Display the image and prediction if the prediction is given
    if pred is not None:
        ax2.imshow(images[0], cmap="gray")
        ax2.imshow(images[2], cmap="gray", alpha=alpha)
        ax2.set_title("Prediction")
        ax2.axis("off")

    plt.show()


def display_decorator(widget: callable):
    """
    Decorator to display a given widget which is returned by the given function.
    Parameters
    ----------
    widget : callable
        The function which returns the widget to be displayed.

    Returns
    -------

    """
    def wrapper(*args, **kwargs):
        inner_widget = widget(*args, **kwargs)
        display(inner_widget)

    return wrapper


@display_decorator
def display_prediction(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = 0.5) -> interactive:
    """
    Display the predicted mask and ground truth over the image.
    Parameters
    ----------
    image : np.ndarray
        The image being displayed.
    gt : np.ndarray
        The ground truth mask.
    pred : np.ndarray
        The predicted mask.
    alpha : float, optional
        The opacity of the masks.

    Returns
    -------
    interactive
        The interactive widget.

    """

    axes = ["xy", "yz", "xz"]

    axes_widget = Dropdown(options=axes)
    slice_widget = IntSlider(min=0, max=image.shape[0] - 1, step=1)

    def update_slice_range(*args):
        slice_max_lst = [image.shape[2] - 1, image.shape[0] - 1, image.shape[1] - 1]
        slice_dict = dict(zip(axes, slice_max_lst))
        slice_widget.max = slice_dict[axes_widget.value]

    slice_widget.observe(update_slice_range, "value")

    widget = interactive(show_prediction,
                         axes=axes_widget,
                         slice_num=slice_widget,
                         image=fixed(image),
                         gt=fixed(gt),
                         pred=fixed(pred),
                         alpha=fixed(alpha))
    return widget
