import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive, fixed, IntSlider, Dropdown, FloatSlider
from IPython.display import display


def show_prediction(axes: str, slice_num: int, image: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = .5):
    """
    Show the predicted mask and ground truth over the image.
    Any of the given images can be set to None to not be displayed.

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

    def validate_img(img, axes, slice_num):
        if axes == "xy":
            return img[:, :, slice_num] if img is not None else None
        elif axes == "yz":
            return img[slice_num, :, :] if img is not None else None
        elif axes == "xz":
            return img[:, slice_num, :] if img is not None else None

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Gets the correct slice from the correct axis
    images = [validate_img(img, axes, slice_num) for img in [image, gt, pred]]

    # Display the image and ground truth if the ground truth is given
    # The image is displayed by default if given
    if images[0] is not None:
        ax1.imshow(images[0], cmap="gray")
        ax2.imshow(images[0], cmap="gray")
    if gt is not None:
        ax1.imshow(images[1], cmap="gray", alpha=alpha)

    # Display the prediction if the prediction is given
    if pred is not None:
        ax2.imshow(images[2], cmap="gray", alpha=alpha)

    ax1.set_title("Ground Truth")
    ax1.axis("off")

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
def display_prediction(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> interactive:
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

    Returns
    -------
    interactive
        The interactive widget.

    """

    axes = ["xy", "yz", "xz"]

    shape = None

    for img in [image, gt, pred]:
        if img is not None:
            shape = img.shape
            break

    axes_widget = Dropdown(options=axes)
    slice_widget = IntSlider(min=0, max=shape[0] - 1, step=1)
    alpha_widget = FloatSlider(min=0, max=1, step=0.1, value=.5)

    def update_slice_range(*args):
        slice_max_lst = [shape[2] - 1, shape[0] - 1, shape[1] - 1]
        slice_dict = dict(zip(axes, slice_max_lst))
        slice_widget.max = slice_dict[axes_widget.value]

    slice_widget.observe(update_slice_range, "value")

    widget = interactive(show_prediction,
                         axes=axes_widget,
                         slice_num=slice_widget,
                         image=fixed(image),
                         gt=fixed(gt),
                         pred=fixed(pred),
                         alpha=alpha_widget)
    return widget
