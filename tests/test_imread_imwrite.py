from cvutils.imsave import imsave
from cvutils import imread, imsave


image_path = 'asset/rick.png'
new_image_path = 'asset/rick_1.png'


def test_imread_imsave():
    image = imread(image_path)
    imsave(image, new_image_path)
