import os
import pandas as pd
import cv2 as cv
import numpy as np


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
        frame = frame.astype('float32')

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
        self.fps = None  # You can set a default fps if needed

    def process_frame(self, frame):
        if self.out_video_writer is None:
            height, width, _ = frame.shape
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            fps = self.fps or 30
            self.out_video_writer = cv.VideoWriter(self.output_path, fourcc, fps, (width, height))

        self.out_video_writer.write(frame)

        # return the frame for potential further processing or display
        return frame

    def release(self):
        if self.out_video_writer:
            self.out_video_writer.release()

class FrameProcessorDisplayer:
    def __init__(self, window_name='processed Video', name='displayer', box='False'):
        self.window_name = window_name
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        self.name = name
        self.bounding_box = bool(box)
        self.detector_Haar =  cv.CascadeClassifier(
                'E:/Project.Pycharm/FaceDetection/Face_detection/Models/haarcascade_frontalface_alt.xml')
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    def plot_rectangle(self, frame, faces):
        # There are 4 values in face array, x,y,h(eight),w(idth)
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
        return frame

    def process_frame(self, frame):
        if self.bounding_box:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.detector_Haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            frame = self.plot_rectangle(frame.copy(), faces)
            cv.imshow(self.window_name, frame)
        else:
            cv.imshow(self.window_name, frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            raise Exception('User interrupted video display.')

        return frame

    def release(self):
        cv.destroyWindow(self.window_name)

# Enhance the contrast of the image
class FrameProcessorHistogramEqualization:
    def __init__(self, name='histogram_equalization'):
        self.name = name

    def process_frame(self, frame):
        img_yuv = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
        img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
        return img_output

# Reduce noise that might interfere with face detection
class FrameProcessorNoiseReduction:
    def __init__(self, name='noise_reduction'):
        self.name = name

    def process_frame(self, frame):
        return cv.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

# Adjust the brightness of the image
class FrameProcessorGammaCorrection:
    def __init__(self, gamma=1.0, name='gamma_correction'):
        self.gamma = gamma
        self.name = name
        self.inv_gamma = 1.0 / gamma
        self.table = np.array([((i / 255.0) ** self.inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    def process_frame(self, frame):
        return cv.LUT(frame, self.table)

# Enhance the edges in the image, making faces more distinguishable
class FrameProcessorSharpening:
    def __init__(self, name='sharpening'):
        self.name = name

    def process_frame(self, frame):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv.filter2D(frame, -1, kernel)