import numpy as np
import cv2


def imsave(image: np.ndarray, path: str) -> None:
    # check data type
    if image.dtype == np.float:
        # check image value range
        image_max = image.max()
        image_min = image.min()
        assert image_max <= 1.0 or image_min >= 0.0, \
            f'image value (float) of out of range, image.max()={image_max},'\
            f' image.min()={image_min}.'
        image = (image * 255).astype(np.uint8)
    else:
        # dtype of image is int
        image_max = image.max()
        image_min = image.min()
        assert image_max <= 255 or image_min >= 0, \
            f'image value (int) of out of range, image.max()={image_max},'\
            f' image.min()={image_min}.'
        image = image.astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)
