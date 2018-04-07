from skimage import color


def rgb_to_lab(images):
    return color.rgb2lab(images)