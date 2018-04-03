# Unroll image data into numpy array
from PIL import Image
import numpy as np
import cv2
from scipy import ndimage
import math


def getBestShift(img):
    """
    This function calculates an image's centroid and the distance needed 
    to shift the centroid to the centre of a img.shape matrix.
    
    Arguments:
    ----------
    img
        An image pixel matrix

    Returns:
    -------
    integer
        Two integers values of the required shift distance in X and Y axis.
    """
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    """
    This function shifts the centroid of an image to the centre of a img.shape matrix.
    
    Arguments:
    ----------
    img - An image pixel matrix
    sx, sy - shift distance in X and Y axis

    Returns:
    -------
    matrix - shifted matrix
    """
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def preprocess_image(filepath):
    """
    This function filters, resizes and centers the image according to its centroid.
    
    Arguments:
    ----------
    filepath : str
        Path to an image file

    Returns:
    -------
    matrix
        A 28x28 matrix containing the processed image pixel values
    """
    #Open greyscale image file as a matrix of pixels 
    gray = Image.open(filepath).convert('L')
    gray = gray.resize((20,20), Image.ANTIALIAS)
    gray = np.array(gray)
    
    #Apply threshold on pixel value to filter the image
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #Remove every row and column at the sides of the image which are completely black
    while np.sum(gray[0]) == 0:
        gray = gray[1:]
    
    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)
    
    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]
    
    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)
        
    rows, cols = gray.shape
    
    #Resize the image to fit a 20x20 pixel box
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))
     
    #Get a 28x28 image by add the missing black rows and columns to the image
    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')
    
    #Find the centroid of the image and align it with the centre of 28x28 box
    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    
    return shifted


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
    im = im.resize((28,28), Image.ANTIALIAS)
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

