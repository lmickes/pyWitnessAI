import numpy as np


# Defining colors used below
colors = {
    'red': (1, 0, 0),
    'yellow': (1, 0.7, 0.1),
    'green': (0, 1, 0),
    'blue': (0, 0, 1),
    'magenta': (1, 0, 1),
    'cyan': (0, 1, 1),
    'white': (1, 1, 1),
    'black': (0, 0, 0),
    'gray': (0.49, 0.49, 0.49),
    'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
    'dark_gray': (0.2, 0.2, 0.2),
    'light_gray': (0.86, 0.86, 0.86)
}

legend_colors = {
    'general': colors['black'],
    'mean': colors['green'],
    'mtcnn': colors['blue'],
    'opencv': colors['red'],
    'fastmtcnn': colors['yellow'],
    'lineup': colors['gray']

}

line_styles = {
    'mtcnn': '-',
    'opencv': '--',
    'lineup': '-.',
    'fastmtcnn': '--',
}

