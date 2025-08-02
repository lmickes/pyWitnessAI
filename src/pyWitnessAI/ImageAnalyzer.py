import os
import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

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
    def __init__(self, column_images, row_images, distance_metric="euclidean",
                 backend="opencv", enforce_detection=False, model="VGG-Face",
                 align=False, normalization="base"):
        """
        Initialize the ImageAnalyzer with ImageLoader instances for columns and rows.
        """
        self.column_images = column_images
        self.row_images = row_images
        self.distance_metric = distance_metric
        self.backend = backend
        self.enforceDetection = enforce_detection
        self.model = model
        self.align = align
        self.normalization = normalization
        self.similarity_matrix = None  # To store the final similarity matrix
        self.method_used = None  # To track which method was used (analyze or process)

        # Initialize MTCNN and InceptionResnetV1 for embedding extraction
        self.mtcnn = MTCNN()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def get_embedding(self, image):
        """
        Obtain the facial embedding for a given image using DeepFace.
        """
        # Convert PIL.Image.Image to NumPy array
        image_array = np.array(image)

        embedding = DeepFace.represent(
            image_array,
            model_name=self.model,
            enforce_detection=self.enforceDetection,
            detector_backend=self.backend,
            align=self.align,
            normalization=self.normalization
        )
        return np.array(embedding[0]['embedding'])

    def get_embedding_facenet(self, img):
        """
        Extract facial embedding using MTCNN and InceptionResnetV1.
        """
        face = self.mtcnn(img)
        if face is None:
            return None
        emb = self.resnet(face.unsqueeze(0))   # Extract embedding
        return emb

    def calculate_similarity_euclidean(self, embedding1, embedding2):
        """
        Calculate L2-normalized Euclidean distance between two embeddings.
        """
        if isinstance(embedding1, torch.Tensor):
            return np.sqrt(((embedding1 - embedding2) * (embedding1 - embedding2)).sum().item())
        else:
            return np.linalg.norm(embedding1 - embedding2)

    def calculate_similarity_euclidean_l2(self, embedding1, embedding2):
        """
        Calculate L2-normalized Euclidean distance between two embeddings.
        """
        if isinstance(embedding1, torch.Tensor):
            embedding1 = embedding1.detach().numpy()
        if isinstance(embedding2, torch.Tensor):
            embedding2 = embedding2.detach().numpy()

        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        return np.linalg.norm(embedding1 - embedding2)

    def calculate_similarity_cosine(self, embedding1, embedding2):
        """
        Calculate cosine distance (1 - cosine similarity) between two embeddings.
        """
        if isinstance(embedding1, torch.Tensor):
            embedding1 = embedding1.detach().numpy()
        if isinstance(embedding2, torch.Tensor):
            embedding2 = embedding2.detach().numpy()
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        return 1 - np.dot(embedding1, embedding2)

    def process_embedding(self):
        """
        Generate similarity matrix using DeepFace.represent (embedding-based method).
        """
        column_embeddings = {}
        row_embeddings = {}

        from contextlib import redirect_stdout, redirect_stderr

        f = open('output.txt', 'w')
        redirect_stdout(f)
        redirect_stderr(f)

        print("Extracting embeddings for column images...")
        for image_base, image in tqdm(self.column_images.images.items()):
        # for image_base, image in self.column_images.images.items():
            column_embeddings[image_base] = self.get_embedding(image)

        print("Extracting embeddings for row images...")
        for image_base, image in tqdm(self.row_images.images.items()):
            row_embeddings[image_base] = self.get_embedding(image)

        print("Calculating similarities...")
        similarity_data = []

        for row_base, row_embedding in tqdm(row_embeddings.items()):
            row_scores = []
            for column_base, column_embedding in column_embeddings.items():
                if self.distance_metric == "euclidean":
                    similarity_score = self.calculate_similarity_euclidean(row_embedding, column_embedding)
                elif self.distance_metric == "cosine":
                    similarity_score = self.calculate_similarity_cosine(row_embedding, column_embedding)
                elif self.distance_metric == "euclidean_l2":
                    similarity_score = self.calculate_similarity_euclidean_l2(row_embedding, column_embedding)
                else:
                    raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
                row_scores.append(similarity_score)
            similarity_data.append(row_scores)

        self.similarity_matrix = pd.DataFrame(
            similarity_data,
            index=row_embeddings.keys(),
            columns=column_embeddings.keys()
        )
        self.method_used = "process_embedding"

    def process_verify(self):
        """
        Generate similarity matrix using DeepFace.verify (verification-based method).
        """
        print("Calculating similarities using DeepFace.verify...")
        similarity_data = []
        for row_base, row_image in tqdm(self.row_images.images.items()):
            row_scores = []
            for column_base, column_image in self.column_images.images.items():
                try:
                    row_image = np.array(row_image)  # Ensure the image is in NumPy format
                    column_image = np.array(column_image)  # Ensure the image is in NumPy format

                    result = DeepFace.verify(
                        row_image,
                        column_image,
                        model_name=self.model,
                        detector_backend=self.backend,
                        enforce_detection=self.enforceDetection,
                        distance_metric=self.distance_metric,
                        align=self.align,
                        normalization=self.normalization
                    )
                    similarity_score = result['distance']  # Use the distance metric from DeepFace.verify
                except ValueError:
                    similarity_score = None  # Handle cases where verification fails
                row_scores.append(similarity_score)
            similarity_data.append(row_scores)

        self.similarity_matrix = pd.DataFrame(
            similarity_data,
            index=self.row_images.images.keys(),
            columns=self.column_images.images.keys()
        )
        self.method_used = "process_verify"

    def process_with_facenet(self):
        """
        Generate similarity matrix using MTCNN and InceptionResnetV1 for embedding extraction.
        """
        column_embeddings = {}
        row_embeddings = {}

        print("Extracting embeddings for column images using Facenet...")
        for image_base, image in tqdm(self.column_images.images.items()):
            column_embeddings[image_base] = self.get_embedding_facenet(image)

        print("Extracting embeddings for row images using Facenet...")
        for image_base, image in tqdm(self.row_images.images.items()):
            row_embeddings[image_base] = self.get_embedding_facenet(image)

        print("Calculating similarities using Facenet embeddings...")
        similarity_data = []

        for row_base, row_embedding in tqdm(row_embeddings.items()):
            row_scores = []
            for column_base, column_embedding in column_embeddings.items():
                if self.distance_metric == "euclidean":
                    similarity_score = self.calculate_similarity_euclidean(row_embedding, column_embedding)
                elif self.distance_metric == "cosine":
                    similarity_score = self.calculate_similarity_cosine(row_embedding, column_embedding)
                else:
                    raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
                row_scores.append(similarity_score)
            similarity_data.append(row_scores)

        self.similarity_matrix = pd.DataFrame(
            similarity_data,
            index=row_embeddings.keys(),
            columns=column_embeddings.keys()
        )
        self.method_used = "process_facenet"

    def dataframe(self):
        """
        Return the similarity matrix.
        """
        return self.similarity_matrix.round(4)

    def save(self, directory='results', label='similarity_matrix'):
        """
        Save the similarity matrix to a CSV file. Creates the directory if it does not exist.
        :param directory: Directory where the file will be saved.
        :param label: Label to include in the filename.
        """
        if self.similarity_matrix is None:
            raise ValueError("No similarity matrix to save. Please run analyze() or process()/dataframe() first.")

        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct the filename
        filename = f"{label}.csv"
        filepath = os.path.join(directory, filename)

        # Save the similarity matrix to a CSV file
        self.similarity_matrix.to_csv(filepath, index=True)
        print(f"Similarity matrix saved to {filepath}.")