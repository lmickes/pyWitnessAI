import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import io
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import pandas as pd
from PIL import Image
import glob
from typing import Dict, Iterable, Tuple


class ImageLoader:
    def __init__(self, images):
        """
        Initialize with:
          - list of paths or glob patterns
          - a directory path
          - a single glob pattern string
        Stores PIL.Image (RGB) objects.
        """
        self.images: Dict[str, Image.Image] = {}
        self.path_to_images: Dict[str, str] = {}

        if isinstance(images, list):
            image_paths = []
            for item in images:
                if '*' in item or '?' in item or '[' in item:
                    image_paths.extend(glob.glob(item))
                else:
                    image_paths.append(item)
        elif isinstance(images, str) and os.path.isdir(images):
            image_paths = self.find_images_in_directory(images)
        elif isinstance(images, str):
            image_paths = self.find_images_glob(images)
        else:
            raise ValueError("Unsupported images input. Provide a list, directory path, or glob pattern.")

        # De-dup & sort for determinism
        image_paths = sorted(set(image_paths))

        for image_path in image_paths:
            # image = cv.imread(image_path)
            # image = np.array(Image.open(image_path))[:, :, 0:3]
            image = Image.open(image_path).convert("RGB")

            if image is None:
                raise ValueError(f"Failed to load image at {image_path}")
            image_base = os.path.splitext(os.path.basename(image_path))[0]
            self.images[image_base] = image
            self.path_to_images[image_base] = image_path

    def find_images_in_directory(self, directory):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

    def find_images_glob(self, pattern):
        return glob.glob(pattern)

    def dataframe(self) -> pd.DataFrame:
        # PIL uses width, height
        sizes: Iterable[Tuple[int, int]] = (img.size for img in self.images.values())
        widths, heights = zip(*sizes) if self.images else ([], [])
        data = {
            'image_base': list(self.images.keys()),
            'image_path': [self.path_to_images[k] for k in self.images.keys()],
            'width': widths,
            'height': heights,
        }
        return pd.DataFrame(data)
