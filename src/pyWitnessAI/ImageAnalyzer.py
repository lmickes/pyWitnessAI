import cv2 as cv
import os
import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm

class ImageLoader:
    def __init__(self, images):
        """
        Initialize the ImageLoader with a list of image paths, a directory path, or a glob pattern.
        Stores loaded images and their paths in dictionaries.
        """
        self.images = {}
        self.path_to_images = {}

        if isinstance(images, list):
            image_paths = images
        elif isinstance(images, str) and os.path.isdir(images):
            image_paths = self.find_images_in_directory(images)
        elif isinstance(images, str):
            image_paths = self.find_images_glob(images)
        else:
            raise ValueError("Unsupported images input. Provide a list, directory path, or glob pattern.")

        for image_path in image_paths:
            image = cv.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image at {image_path}")
            image_base = os.path.splitext(os.path.basename(image_path))[0]
            self.images[image_base] = image
            self.path_to_images[image_base] = image_path

    def find_images_in_directory(self, directory):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

    def find_images_glob(self, pattern):
        import glob
        return glob.glob(pattern)

    def dataframe(self):
        image_size_x = [image.shape[0] for image in self.images.values()]
        image_size_y = [image.shape[1] for image in self.images.values()]

        data = {
            'image_base': list(self.images.keys()),
            'image_path': list(self.path_to_images.values()),
            'image_size_x': image_size_x,
            'image_size_y': image_size_y
        }
        return pd.DataFrame(data)

class ImageAnalyzer:
    def __init__(self, column_images, row_images, backend, detector="mtcnn", enforceDetection=False, model="Facenet"):
        """
        Initialize the ImageAnalyzer with ImageLoader instances for columns and rows.
        """
        self.column_images = column_images
        self.row_images = row_images
        self.backend = backend
        self.detector = detector
        self.enforceDetection = enforceDetection
        self.model = model

    def get_embedding(self, image):
        """
        Obtain the facial embedding for a given image using DeepFace.
        """
        embedding = DeepFace.represent(
            image,
            model_name=self.model,
            enforce_detection=self.enforceDetection,
            detector_backend=self.detector
        )
        return np.array(embedding[0]['embedding'])

    def calculate_similarity_euclidean(self, embedding1, embedding2):
        """
        Calculate L2-normalized Euclidean distance between two embeddings.
        """
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        return np.linalg.norm(emb1_norm - emb2_norm)

    def calculate_similarity_cosine(self, embedding1, embedding2):
        """
        Calculate cosine distance (1 - cosine similarity) between two embeddings.
        """
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        return 1 - np.dot(emb1_norm, emb2_norm)

    def dataframe(self):
        """
        Construct a long-format DataFrame containing all pairwise similarities between column and row images.
        """
        records = []

        column_embeddings = {}
        row_embeddings = {}

        print("Extracting embeddings for column images...")
        for image_base, image in tqdm(self.column_images.images.items()):
            column_embeddings[image_base] = self.get_embedding(image)

        print("Extracting embeddings for row images...")
        for image_base, image in tqdm(self.row_images.images.items()):
            row_embeddings[image_base] = self.get_embedding(image)

        print("Calculating similarities...")
        for column_base, column_embedding in tqdm(column_embeddings.items()):
            for row_base, row_embedding in row_embeddings.items():
                similarity_euclidean = self.calculate_similarity_euclidean(column_embedding, row_embedding)
                similarity_cosine = self.calculate_similarity_cosine(column_embedding, row_embedding)

                records.append({
                    'column_image_base': column_base,
                    'row_image_base': row_base,
                    'similarity_euclidean': similarity_euclidean,
                    'similarity_cosine': similarity_cosine
                })

        df = pd.DataFrame(records)
        return df

    def analyze(self):
        """
        Generate and return the similarity DataFrame.
        """
        return self.dataframe()
