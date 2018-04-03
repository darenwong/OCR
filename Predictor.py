from ProcessImages import *
from nnlib import *
import matplotlib.pyplot as plt


def predictor(filepath, parameters):
    """
    Parameters
    ----------
    filepath : str
        Path to an image file

    Returns
    -------
    int
        An integer that indicates the ML prediction
    """
    img = preprocess_image(filepath)
    plt.imshow(img/255, cmap='gray')
    plt.colorbar()
    plt.show()
    prediction = predict(parameters, img.reshape((784,1))/255)
    print("It's {}!".format(prediction))
    return prediction

