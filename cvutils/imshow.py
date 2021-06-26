import numpy as np
import matplotlib.pyplot as plt


def imshow_gray(image: np.ndarray) -> None:
    plt.imshow(image, cmap='gray')
    plt.show()


def imshow_rgb(image: np.ndarray) -> None:
    image_shape = image.shape

    # pytorch image
    if image_shape[0] == 3:
        image = image.transpose(1, 2, 0)

    plt.imshow(image)
    plt.show()


def imshow(image: np.ndarray) -> None:
    image = np.squeeze(image)

    if image.ndim == 2:
        imshow_gray(image)
    elif image.ndim == 3:
        imshow_rgb(image)
    else:
        raise ValueError(
            f'The dimension ({image.ndim}) of image is invalid.')
