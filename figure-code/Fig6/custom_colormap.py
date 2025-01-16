import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def custom_cmap(base_color, light_color, dark_color, reverse=False):
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [light_color, base_color, dark_color])

    if reverse:
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [dark_color, base_color, light_color])

    return custom_cmap