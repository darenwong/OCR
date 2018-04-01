def predictor(filepath):
    """
    Parameters
    ----------
    filepath : str
        Path to an image file

    Returns
    -------
    int
        An integer that indicates the ML prediction
    
    plot
        Resize image and plot the image
    """
    (width, height), greyscale_map = image2pixelarray(filepath)
    image_display(filepath)
    prediction = predict(parameters, greyscale_map)
    return print("I think it's a {}!".format(prediction))