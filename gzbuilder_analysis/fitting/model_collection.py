import pandas as pd
import numpy as np
from copy import deepcopy
from gzbuilder_analysis.parsing import sanitize_model
import gzbuilder_analysis.rendering as rg
from gzbuilder_analysis.rendering.sersic import oversampled_sersic_component
from gzbuilder_analysis.rendering.spiral import spiral_arm
from gzbuilder_analysis.config import FIT_PARAMS
from gzbuilder_analysis.fitting import Model


# utility for
class ModelCollection():
    def __init__(self, models):
        self.models = models
        self.data = []
        self.masks = []
        self.sigma_images = []
        self.
