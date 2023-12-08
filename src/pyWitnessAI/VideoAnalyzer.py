import os
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from .Constants import legend_colors, line_styles
from keras.models import load_model
from importlib.resources import files
from .DataFlattener import *
# You should also load the path of cascade, similarity_model, lineup_images before using the analyzer


class VideoAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        self.frame_count = []
        self.frame_width = None
        self.frame_height = None
        self.frame_area = None
        self.average_pixel_values = []
        self.average_value = 0  # This is for the mean of the average pixel values of the whole video

        self.frame_processor = {}
        self.frame_analyzer = {}
        self.frame_analyzer_output = {}
        self.frame_analyzed = 0
        self.frame_total = 0

    def add_analyzer(self, analyzer):
        #  Add an external frame analyzer
        self.frame_analyzer[analyzer.name] = analyzer
        self.frame_analyzer_output[analyzer.name] = []

    def add_processor(self, processor):
        self.frame_processor[processor.name] = processor

    def release_resources(self):
        self.cap.release()
        for processor in self.frame_processor.values():
            if hasattr(processor, 'release'):
                processor.release()

    def get_frame_info(self):
        #  Retrieve frame information
        self.frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_area = self.frame_width * self.frame_height

    def process_video(self, frame_start=0, frame_end=1000000000):
        #  Process the video frame between frame_start and frame_end
        frame_analyzed = 0

        for frame_count in range(frame_start, frame_end + 1):
            ret, frame = self.cap.read()

            if not ret:
                break

            if frame_count == frame_start:
                self.frame_total = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
                self.get_frame_info()

            self.frame_count.append(frame_count)
            average_pixel_value = int(frame.mean())
            self.average_pixel_values.append(average_pixel_value)

            for k in self.frame_processor:
                frame = self.frame_processor[k].process_frame(frame)

            for k in self.frame_analyzer:
                self.frame_analyzer_output[k].append(self.frame_analyzer[k].analyze_frame(frame))

            frame_analyzed += 1

        self.average_value = np.mean(self.average_pixel_values)
        self.frame_analyzed = frame_analyzed
        self.release_resources()
        cv.destroyAllWindows()

    def get_analysis_info(self):
        #  Get the number of analyzed frame and total frames
        return {
            'frame_analyzed': self.frame_analyzed,
            'frame_total': self.frame_total
        }

    def run(self, frame_start=0, frame_end=100000):
        self.process_video(frame_start, frame_end)

    def plot_face_counts(self):
        #  Plots the number of faces against frame numbers
        for k, output in self.frame_analyzer_output.items():
            if k == "lineup":
                continue
            # print(k)  # mtcnn, opencv, similarity
            # print(output)  #  Their output
            face_counts = []
            for data in output:
                #  print("Data:", data)
                if 'face_count' in data:
                    face_counts.append(data['face_count'])
                else:
                    face_counts.append(0)
            plt.plot(self.frame_count, face_counts, label=k, linestyle=line_styles[k], color=legend_colors[k])

        plt.xlabel('Frame')
        plt.ylim(0, 5)
        plt.ylabel('Number of Faces')
        plt.title('Number of Faces vs Frame Number')
        plt.legend()
        plt.grid(True)

    def plot_face_areas(self):
        #  Plots the face area recognized by the classifiers against frame numbers
        for k, output in self.frame_analyzer_output.items():
            if k == "lineup":
                continue
            face_areas = []
            for data in output:
                if 'face_area' in data:
                    face_areas.append(data['face_area'] / self.frame_area)
                else:
                    face_areas.append(0)
            plt.plot(self.frame_count, face_areas, label=k, linestyle=line_styles[k], color=legend_colors[k])

        plt.xlabel('Frame')
        plt.ylim(0, 0.5)
        plt.ylabel('Face Area Ratio')
        plt.title('Face Area Ratio vs Frame Number')
        plt.legend()
        plt.grid(True)

    def plot_average_pixel_values(self):
        #  Plot the average pixel values of the video
        plt.plot(self.frame_count, self.average_pixel_values, color=legend_colors['general'])
        plt.axhline(y=self.average_value, color=legend_colors['mean'], linestyle='--', label='Average value')
        plt.xlabel('Frame')
        plt.ylim(self.average_value-50, self.average_value+50)
        plt.ylabel('Average pixel value')
        plt.title('Pixel Intensity Trend across the Video')
        plt.legend()
        plt.grid(True)

    def plot_mtcnn_confidence_histogram(self):
        #  Plot the confidence histogram of mtcnn
        if 'mtcnn' in self.frame_analyzer_output:
            confidences = []
            for data in self.frame_analyzer_output['mtcnn']:
                if 'confidence' in data:
                    confidences.append(data['confidence'])

            flattened_confidences = []
            for sublist in confidences:
                for item in sublist:
                    flattened_confidences.append(item)

            plt.hist(flattened_confidences, bins=30, edgecolor='k')
            plt.xlabel('confidence')
            plt.ylabel('Frequency')
            plt.title('MTCNN Confidence Histogram')
            plt.grid(True)

        else:
            print('MTCNN data are not found in the output.')

    def save_data(self, directory='results', prefix='analyzed'):
        #  Save the analyzed data results to .csv files.
        if not os.path.exists(directory):
            os.makedirs(directory)

        data = {
            'frame': self.frame_count,
            'avg_pixel_value': self.average_pixel_values
        }

        for analyzer_name, results_list in self.frame_analyzer_output.items():
            # if all entries are dictionaries
            if all(isinstance(entry, dict) for entry in results_list):
                for key in results_list[0].keys():
                    data[f'{analyzer_name}_{key}'] = [entry[key] for entry in results_list]
                    #print(1)
            else:
                # if the results list contains non-dictionary entries (like lists)
                # If the entry is a list, save its length (For simplicity).
                data[f'{analyzer_name}_length'] = [len(entry) if isinstance(entry, list) else None for entry in
                                                   results_list]

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(directory, f'{prefix}_data.csv'), index=False)

    def save_data_flattened(self, directory='results', prefix='analyzed_flattened'):
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Prepare the basic data with frame count and average pixel values
        data = {
            'frame': self.frame_count,
            'avg_pixel_value': self.average_pixel_values
        }

        # Initialize a set to keep track of all keys
        all_keys = {}

        # Find all the keys
        for analyzer_name, results_list in self.frame_analyzer_output.items():
            # Flatten each entry in the results list
            for result in results_list:
                flattened_keys = flatten_keys(result)
                for key in flattened_keys:
                    all_keys[key] = []

        # Match data with the corresponding keys
        for analyzer_name, results_list in self.frame_analyzer_output.items():
            for result in results_list:
                flattened_keys = flatten_keys(result)
                flattened_values = flatten_data(result)
                for key in all_keys:
                    if key in flattened_keys:
                        all_keys[key].append(flattened_values[flattened_keys.index(key)])
                    else:
                        all_keys[key].append(0)

        #

        # Create a DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(directory, f'{prefix}_data_flattened.csv'), index=False)


class FrameProcessorCropper:
    def __init__(self, x1, x2, y1, y2, name='cropper'):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.name = name

    def process_frame(self, frame):
        #  Crop the frame using the provided coordinates
        cropped_frame = frame[self.y1:self.y2, self.x1:self.x2]

        return cropped_frame


class FrameProcessorMonochrome:
    def __init__(self, name='monochrome'):
        self.name = name

    def process_frame(self, frame):
        #  Convert frame to grayscale
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #  Convert the grayscale frame back to BGR for video writer and other stuffs
        bgr_frame = cv.cvtColor(gray_frame, cv.COLOR_GRAY2BGR)

        return bgr_frame


class FrameProcessorNormalizer:
    def __init__(self, set_min, set_max, name='normalizer'):
        self.set_min = set_min
        self.set_max = set_max
        self.name = name

    def process_frame(self, frame):
        frame_min = frame.min()
        frame_max = frame.max()
        set_range = self.set_max - self.set_min

        #  Convert the frame to float for computation
        frame = frame.astaype('float32')

        #  Create a matrix filled with minimum pixel value of the frame
        matrix_frame_min = np.full(frame.shape, frame_min, dtype='float32')

        numerator = cv.subtract(frame, matrix_frame_min)
        denominator = frame_max - frame_min

        if denominator == 0:
            denominator = 1

        normalized_frame = set_range * (numerator / denominator) + self.set_min

        return normalized_frame.astype('uint8')


class FrameProcessorRemover:
    def __init__(self, boxes, name='remover'):
        self.boxes = boxes  # Each box is a tuple (x1, x2, y1, y2)
        self.color_gray = (128, 128, 128)
        self.name = name

    def process_frame(self, frame):
        # Replace specified rectangular area with gray color
        height, width, _ = frame.shape

        for (x1, x2, y1, y2) in self.boxes:
            #  Ensure the rectangle coordinates are within frame dimensions
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            frame[y1:y2, x1:x2] = self.color_gray

        return frame


class FrameProcessorVideoWriter:
    def __init__(self, output_path, name='video_writer'):
        self.output_path = output_path
        self.out_video_writer = None
        self.name = name

    def process_frame(self, frame):
        if self.out_video_writer is None:
            height, width, _ = frame.shape
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            fps = 30
            self.out_video_writer = cv.VideoWriter(self.output_path, fourcc, fps, (width, height))

        self.out_video_writer.write(frame)

        # return the frame for potential further processing or display
        return frame

    def release(self):
        if self.out_video_writer:
            self.out_video_writer.release()


class FrameProcessorDisplayer:
    def __init__(self, window_name='processed Video', name='displayer'):
        self.window_name = window_name
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        self.name = name

    def process_video(self, frame):
        cv.imshow(self.window_name, frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            raise Exception('User interrupted video display.')

        return frame

    def release(self):
        cv.destroyWindow(self.window_name)


class FrameAnalyzerMTCNN:
    def __init__(self, name="mtcnn"):
        self.detector = MTCNN()
        self.name = name

    def analyze_frame(self, frame):
        faces = self.detector.detect_faces(frame)
        confidence = self.get_confidence(faces)
        face_count = len(faces)
        face_area = self.get_face_area(faces)

        return {
            f'{self.name}_face_count': face_count,
            f'{self.name}_face_area': face_area,
            f'{self.name}_confidence': confidence
        }

    def get_confidence(self, faces):
        confidence = []

        for face in faces:
            confidence.append(face['confidence'])

        return confidence

    def get_face_area(self, faces):
        face_area_sum = sum(face['box'][2] * face['box'][3] for face in faces)
        return face_area_sum


class FrameAnalyzerOpenCV:
    def __init__(self,
                 cascade_path=str(files("pyWitnessAI.OpenCV_Models").joinpath("haarcascade_frontalface_alt.xml")),
                 name='opencv'):
        self.face_cascade = cv.CascadeClassifier(cascade_path)
        self.name = name

    def analyze_frame(self, frame):
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        face_count = len(faces)
        face_area = self.get_face_area(faces)

        return {
            f'{self.name}_face_count': face_count,
            f'{self.name}_face_area': face_area
        }

    def get_face_area(self, faces):
        face_area_sum = sum([w * h for (x, y, w, h) in faces])
        return face_area_sum


class SimilarityAnalyzer:
    def __init__(self, lineup_images, face_detector="mtcnn", name='similarity'):
        self.face_detector = face_detector
        self.lineup_images = lineup_images  # Pre-processed to a specific size (say, 160x160)
        self.model_path = str(files("pyWitnessAI.FaceNet_Models").joinpath("FACE-DETECT.h5"))
        self.model = load_model(self.model_path)
        self.name = name

        if self.face_detector == "mtcnn":
            self.detector = MTCNN()
        elif self.face_detector == "opencv":
            cascade_path = \
                str(files("pyWitnessAI.OpenCV_Models").joinpath("haarcascade_frontalface_alt.xml"))
            self.face_cascade = cv.CascadeClassifier(cascade_path)

        # Pre-compute embeddings for the provided lineup images
        self.lineup_embeddings = [self.get_embedding(image) for image in self.lineup_images]

    def analyze_frame(self, frame):
        # if self.face_detector == "mtcnn":    # If there are more face detectors
        faces = self.detector.detect_faces(frame)
        # extract detected face regions
        face_images = [frame[y:y + h, x:x + w]
                       for (x, y, w, h) in [face['box'] for face in faces]]

        # compute embeddings for each detected face in the frame
        frame_embeddings = [self.get_embedding(face) for face in face_images]

        # compute similarity values
        similarity_values = []
        for frame_emb in frame_embeddings:
            similarities = [self.calculate_similarity(frame_emb, lineup_emb)
                            for lineup_emb in self.lineup_embeddings]
            similarity_values.append(similarities)
            # print(f"similarities: {similarity_values}")

        return {f'{self.name}_values': similarity_values}

    def get_embedding(self, face_pixels):
        # Ensure image is of the right size
        face_pixels = cv.resize(face_pixels, (100, 100))  # Assuming 160x160 is the expected input size
        # Convert RGB image to grayscale
        face_pixels = cv.cvtColor(face_pixels, cv.COLOR_BGR2GRAY)
        # Convert pixel values to float (if using normalization)
        face_pixels = face_pixels.astype('float32')
        # Possibly normalize pixel values (e.g., mean-center and scale)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # Expand dimensions to one sample and add channel dimension
        face_pixels = np.expand_dims(face_pixels, axis=0)
        face_pixels = np.expand_dims(face_pixels, axis=-1)  # Adding channel dimension
        # Predict using the FaceNet model
        embedding = self.model.predict(face_pixels)
        return embedding[0]

    def calculate_similarity(self, emb1, emb2):
        #  return np.linalg.norm(emb1 - emb2) #  L2 norm
        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        similarity = dot_product / (norm_emb1 * norm_emb2)
        return similarity


class LineupLoader:
    def __init__(self, image_number=0, image_path=None):
        if image_path is None:
            self.image_path = ["E:/Project.Pycharm/FaceDetection/Assessment/Lineup/Damien99.jpg",
                               "E:/Project.Pycharm/FaceDetection/Assessment/Lineup/Damien1999.jpg"]
        else:
            self.image_path = image_path
        if image_number == 0:
            self.number = len(image_path)
        else:
            self.number = image_number
        self.lineup_images = []

    def preprocess_image(self, image, target_size=(160, 160)):
        image = cv.resize(image, target_size)
        return image

    def load_image(self):
        count = 0
        loaded_images = []

        for path in self.image_path:
            if count >= self.number:
                break
            image = cv.imread(path)
            processed_image = self.preprocess_image(image)
            loaded_images.append(processed_image)
            count += 1
        self.lineup_images = loaded_images
        return loaded_images



















