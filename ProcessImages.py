# Unroll image data into numpy array
import Image
import numpy as np

def image2pixelarray(filepath):
    """
    Parameters
    ----------
    filepath : str
        Path to an image file

    Returns
    -------
    tuple
        A tuple consisting of (width, height)
    list
        A list of lists which make it simple to access the greyscale value by
        im[y][x]
    """
    im = Image.open(filepath).convert('L')
    (width, height) = im.size
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map)
    return (width, height), greyscale_map

image2pixelarray()