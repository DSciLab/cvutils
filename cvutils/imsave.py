import numpy as np
import cv2


def imsave(image: np.ndarray, path: str) -> None:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)
