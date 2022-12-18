from typing import Union
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, BatchNormalization, \
    Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from .loss import dice_loss, dice_coef


def unet_3d(input_shape: tuple = (128, 128, 128, 1),
            learning_rate: float = 1e-4,
            num_classes: int = 2,
            weight_decay: float = 1e-4,
            depth: int = 4,
            pool_size: tuple = (2, 2, 2),
            n_base_filters: int = 16,
            loss: callable = dice_loss,
            metrics: callable = dice_coef):
    """
    Create a 3D U-Net model

    Parameters
    ----------
    input_shape : tuple
        Shape of the input image
    learning_rate : float
        Learning rate for the optimizer
    num_classes : int
        Number of classes
    weight_decay : float
        Weight decay for the optimizer
    depth : int
        Depth of the U-Net
    pool_size : tuple
        Pool size for the max pooling operations
    n_base_filters : int
        Number of filters in the first convolutional layer
    loss : callable, optional
        Loss function
    metrics : callable, optional
        Metrics to evaluate the model

    Returns
    -------
    model : Model
        The 3D U-Net model

    """

    # Build U-Net model
    inputs = Input(input_shape)
    current_layer = inputs

    # Add encoding layers with max pooling
    encoder_layers = []
    for layer_depth in range(depth):

        # Add convolutional layers
        layer1 = Conv3D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3, 3), padding="same",
                        kernel_regularizer=regularizers.l2(weight_decay))(current_layer)

        # Add batch normalization
        layer1 = BatchNormalization()(layer1)

        # Add activation function
        layer1 = Activation("relu")(layer1)
        layer2 = Conv3D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3, 3), padding="same",
                        kernel_regularizer=regularizers.l2(weight_decay))(layer1)
        layer2 = BatchNormalization()(layer2)
        layer2 = Activation("relu")(layer2)
        if layer_depth < depth - 1:

            # Add max pooling
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            encoder_layers.append(layer2)
        else:
            current_layer = layer2

    # Add decoding layers with upsampling
    for layer_depth in range(depth - 2, -1, -1):
        layer1 = Conv3D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3, 3), padding="same",
                        kernel_regularizer=regularizers.l2(weight_decay))(current_layer)
        layer1 = BatchNormalization()(layer1)
        layer1 = Activation("relu")(layer1)
        layer2 = Conv3D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3, 3), padding="same",
                        kernel_regularizer=regularizers.l2(weight_decay))(layer1)
        layer2 = BatchNormalization()(layer2)
        layer2 = Activation("relu")(layer2)

        # Add upsampling
        current_layer = UpSampling3D(size=pool_size)(layer2)

        # Add concatenation
        current_layer = concatenate([encoder_layers[layer_depth], current_layer], axis=4)

    # Add a dense layer with sigmoid activation
    final_layer = Conv3D(filters=num_classes, kernel_size=(1, 1, 1), activation="sigmoid")(current_layer)

    # Compile the model
    model = Model(inputs=inputs, outputs=final_layer)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss,
                  metrics=metrics
                  )

    return model
