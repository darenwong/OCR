# Unroll image data into numpy array
from PIL import Image
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
    im = im.resize((20,20), Image.ANTIALIAS)
    (width, height) = im.size
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map)/np.amax(greyscale_map)
    greyscale_map = np.array([greyscale_map]).T
    return (width, height), greyscale_map


def image_display(filepath):
    """
    Parameters
    ----------
    filepath : str
        Path to an image file

    Returns
    -------
    plot
        Resize image and plot the image
    """
    (width, height), greyscale_map = image2pixelarray(filepath)
    img = greyscale_map.reshape(width,height)
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    return plt.show()

