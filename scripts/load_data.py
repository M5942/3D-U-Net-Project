import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
from .preprocess import load_nifti, preprocess_image, preprocess_mask, get_patches

SEED = 42


def get_data(data_dir: str, patch_size: tuple = (64, 64, 64), test_size: float = 0.3):

    # Get the folders in the directory
    image_dir = os.path.join(data_dir, 'image')
    label_dir = os.path.join(data_dir, 'label')

    # Get the images and masks
    image_paths = [os.path.join(image_dir, image) for image in os.listdir(image_dir)]
    label_paths = [os.path.join(label_dir, label) for label in os.listdir(label_dir)]

    # Split the data into training and testing
    image_paths_train, image_paths_test, label_paths_train, label_paths_test = \
        train_test_split(image_paths, label_paths, test_size=test_size, random_state=SEED)

    # Prints the test images for prediction later on
    print("Test Images Paths: ")
    print(*image_paths_test, sep='\n')

    # Load the images and masks
    X_train = np.array(list(map(load_nifti, image_paths_train)), dtype=object)
    y_train = np.array(list(map(load_nifti, label_paths_train)), dtype=object)
    X_test = np.array(list(map(load_nifti, image_paths_test)), dtype=object)
    y_test = np.array(list(map(load_nifti, label_paths_test)), dtype=object)

    # Get new dimensions
    new_dims = np.round(np.array(X_train[0].shape) / patch_size[0]) * patch_size[0]

    # Preprocess the data
    X_train = np.array([preprocess_image(image, new_dims) for image in X_train])
    y_train = np.array([preprocess_mask(image, new_dims) for image in y_train])
    X_test = np.array([preprocess_image(image, new_dims) for image in X_test])
    y_test = np.array([preprocess_mask(image, new_dims) for image in y_test])

    # Get the patches
    X_train = np.array([get_patches(image, patch_size) for image in X_train])
    y_train = np.array([get_patches(image, patch_size) for image in y_train])
    X_test = np.array([get_patches(image, patch_size) for image in X_test])
    y_test = np.array([get_patches(image, patch_size) for image in y_test])

    # Reshape the data to have shape (patch_num, patch_size, patch_size, patch_size, channels)
    # to_categorical converts the labels to binary class matrices
    X_train = np.expand_dims(np.vstack(X_train), axis=-1)
    y_train = to_categorical(np.vstack(y_train))
    X_test = np.expand_dims(np.vstack(X_test), axis=-1)
    y_test = to_categorical(np.vstack(y_test))

    # Shuffle the patches in unison
    def unison_shuffled_copies(a, b):
        if len(a) != len(b):
            raise AssertionError
        # Keeps the shuffling consistent to get reproducible results
        p = np.random.RandomState(SEED).permutation(len(a))
        return a[p], b[p]

    X_train, y_train = unison_shuffled_copies(X_train, y_train)
    X_test, y_test = unison_shuffled_copies(X_test, y_test)

    return X_train, X_test, y_train.astype(np.float64), y_test.astype(np.float64)






