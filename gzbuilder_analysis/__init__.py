import json

# High level functions (to be moved to relevant folders upon creation)

# parsing of models
def parse_classification(classification, image_size):
    """Extract a more usable model from a zooniverse annotation
    """
    annotation = json.loads(classification['annotations'])
    return parse_annotation(annotation, image_size)

def parse_annotation(annotation, image_size, scale, wcs=None):
    """Extract a more usable model from a zooniverse annotation
    """
    pass


def scale_model(model, scale):
    """Scale a model to represent different image sizes
    """
    pass


def reproject_model(model, wcs_in, wcs_out):
    """Rotate, translate and scale model from one WCS to another
    """
    pass


# aggregation of models
def cluster_components(models):
    """Accepts a list of models and returns clusters of disks, bulges, bars and
    spiral arms
    """
    pass


def aggregate_components(clustered_components):
    """Accepts clusters of components and constructs an aggregate model
    """
    pass


# rendering of model
def render_model(model, image_size, psf):
    """Render a PSF-convolved model
    """
    pass


# fitting
def chisq(rendered_model, image_data, sigma_image, pixel_mask=None):
    """Accepts a rendered model, a masked image array (numpy.ma.core.MaskedArray)
    and a sigma image (also numpy.ma.core.MaskedArray), and returns a reduced
    chi-squared value
    """
    pass
