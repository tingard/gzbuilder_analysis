class GalaxyBuilderError(Exception):
    """Base class for exceptions in this module."""
    pass


class InvalidModelError(GalaxyBuilderError):
    """Exception raised for errors in the input model.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
