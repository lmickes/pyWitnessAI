import cv2 as cv
import os

class ImageLoader:
    def __init__(self, images):

        self.images = {}
        self.path_to_images = {}
        if type(images) is list:
            image_paths = images
        elif type(images) is str and os.path.isdir(images):
            image_paths = self.find_images_in_directory(images)
        elif type(images) is str:
            image_paths = self.find_images_glob(images)

        for image_path in image_paths:
            image = cv.imread(image_path)
            image_base = os.path.splitext(os.path.basename(image_path))[0]
            self.images[image_base] = image
            self.path_to_images[image_base] = image_path


    def find_images_in_directory(self, directory):
        import os
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

    def find_images_glob(self, pattern):
        import glob
        return glob.glob(pattern)

    def dataframe(self):
        import pandas as pd

        image_size_x = [image.shape[0] for image in self.images.values()]
        image_size_y = [image.shape[1] for image in self.images.values()]

        data = {'image_base': list(self.images.keys()),
                'image_path': list(self.path_to_images.values()),
                'image_size_x': image_size_x,
                'image_size_y': image_size_y}
        return pd.DataFrame(data)

class ImageAnalyzer:
    def __init__(self):
        pass




