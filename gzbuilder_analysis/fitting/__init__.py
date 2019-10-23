import numpy as np

def chisq(model_data, galaxy_data, sigma_image, pixel_mask=None):
    # chisq = 1/N_dof * sum(f_data(x, y) - f_model(x, y))^2 / sigma(x, y)^2
    # Assume N_dof ~ number of unmasked pixels
    # pixel_mask values of True means a pixel should be masked
    if type(galaxy_data) != np.ma.core.MaskedArray:
        galaxy_data = np.ma.masked_array(galaxy_data, pixel_mask)
    return 1 / len(galaxy_data.compressed()) * np.sum(
        ((model_data - galaxy_data) / sigma_image)**2
    )
