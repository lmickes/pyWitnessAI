import os
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from .Constants import legend_colors, line_styles, get_color_for_analyzer, get_style_for_analyzer
from keras.models import load_model
from importlib.resources import files
from .DataFlattener import *
from .PhotoAssigner import *
from PIL import Image
import heapq
from deepface import DeepFace
import time
# You should also load the path of cascade, similarity_model, lineup_images before using the analyzer

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class VideoAnalyzer:
    def __init__(self, video_path, save_directory='Video analysis results'):
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

        self.save_directory = save_directory
        self.top_frames = None  # An attribute to get the best quality frame
        self.find_probe_frames_detector = None
        self.find_probe_frames_method = None

        self.face_gallery = {}  # {label: {"rep": np.ndarray, "samples": [(frame_idx, (x,y,w,h))], "thumb": [np.ndarray,...]}}
        self.face_labels_by_frame = []  # Corresponding to frame_count, list of lists of labels per frame, in order
        self._gallery_built_from = None  # Model used to build the gallery (e.g., 'mtcnn')
        self._gallery_model_name = None  # Model name used for embeddings (e.g., 'Facenet512')

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

        # Initialize timing dictionary
        analyzer_timings = {analyzer: 0.0 for analyzer in self.frame_analyzer}

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

            for analyzer_name, analyzer in self.frame_analyzer.items():
                start_time = time.time()  # Record start time
                self.frame_analyzer_output[analyzer_name].append(analyzer.analyze_frame(frame))
                end_time = time.time()  # Record end time
                analyzer_timings[analyzer_name] += end_time - start_time  # Accumulate time

            frame_analyzed += 1

        self.average_value = np.mean(self.average_pixel_values)
        self.frame_analyzed = frame_analyzed
        self.release_resources()
        cv.destroyAllWindows()

        # Print timing results
        for analyzer_name, total_time in analyzer_timings.items():
            print(f"Total time for {analyzer_name}: {total_time:.2f} seconds")

    def get_analysis_info(self):
        #  Get the number of analyzed frame and total frames
        return {
            'frame_analyzed': self.frame_analyzed,
            'frame_total': self.frame_total
        }

    def run(self, frame_start=0, frame_end=100000):
        self.process_video(frame_start, frame_end)

    def build_face_gallery(self, detector='mtcnn', model_name='Facenet512',
                           max_distance=0.90, save_dir=None, max_samples_per_label=4):
        """
        Identify and cluster faces across all frames in the video. Label them as face1, face2, etc.

        Parameters
        ----------
        detector: Choose which detector's output to use for cropping faces
        model_name: DeepFace embedding
        max_distance: The maximum distance threshold to consider two faces as the same person
        save_dir: optional path to save the face gallery
        max_samples_per_label: Each label will store up to this many face thumbnails
        """
        if detector not in self.frame_analyzer_output:
            raise ValueError(f"Detector '{detector}' analyzer is not added or has no output.")

        outputs = self.frame_analyzer_output[detector]
        if len(outputs) != len(self.frame_count):
            raise RuntimeError("Analyzer output length mismatch with frame_count. Did you run process_video()?")

        self.face_gallery.clear()
        self.face_labels_by_frame = [[] for _ in self.frame_count]
        self._gallery_built_from = detector
        self._gallery_model_name = model_name

        # Reopen video to read frames
        cap2 = cv.VideoCapture(self.video_path)
        if not cap2.isOpened():
            raise RuntimeError(f"Cannot reopen video: {self.video_path}")

        current_frame_idx = -1
        label_count = 0

        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            current_frame_idx += 1
            if current_frame_idx >= len(outputs):
                break

            coords = outputs[current_frame_idx].get('coordinates', [])
            if not coords:
                self.face_labels_by_frame[current_frame_idx] = []
                continue

            per_frame_labels = []
            for box in coords:
                face_img = self._crop_face_from_frame(frame, box)
                try:
                    emb = self._represent_face(face_img, model_name=model_name)
                except Exception:
                    # embedding failed, skip this face
                    per_frame_labels.append(None)
                    continue

                # Find the closest existing label
                best_label = None
                best_dist = float('inf')
                for lbl, info in self.face_gallery.items():
                    centroid = info['rep']
                    dist = self._euclidean_distance(emb, centroid)
                    if dist < best_dist:
                        best_dist = dist
                        best_label = lbl

                # Make decision based on distance
                if best_dist <= max_distance and best_label is not None:
                    info = self.face_gallery[best_label]
                    # Update centroid and samples
                    new_centroid = self._update_centroid(info['rep'], emb, len(info['samples']))
                    info['rep'] = new_centroid
                    info['samples'].append((current_frame_idx, tuple(map(int, box))))
                    # Add thumbnail if under limit
                    if len(info['thumb']) < max_samples_per_label:
                        thumb = cv.resize(face_img, (112, 112))
                        info['thumb'].append(thumb)
                    per_frame_labels.append(best_label)
                else:
                    # Create new label
                    label_count += 1
                    new_label = f"face{label_count}"
                    self.face_gallery[new_label] = {
                        "rep": emb,
                        "samples": [(current_frame_idx, tuple(map(int, box)))],
                        "thumb": [cv.resize(face_img, (112, 112))]
                    }
                    per_frame_labels.append(new_label)

            self.face_labels_by_frame[current_frame_idx] = per_frame_labels

        cap2.release()

        # Optionally save the face gallery
        if save_dir:
            self.save_face_gallery(save_dir)

        return self.face_gallery

    def list_face_labels(self):
        # List all identified face labels in the gallery
        return list(self.face_gallery.keys())

    def save_face_gallery(self, save_dir):
        """
        Save the face gallery to the specified directory (label, samples_count, first_seen_frame).
        """
        os.makedirs(save_dir, exist_ok=True)
        # Save thumbnails
        for lbl, info in self.face_gallery.items():
            thumbs = info.get("thumb", [])
            for i, t in enumerate(thumbs):
                path = os.path.join(save_dir, f"{lbl}_{i+1}.jpg")
                cv.imwrite(path, t)

        # Save index CSV
        rows = []
        for lbl, info in self.face_gallery.items():
            samples = info.get("samples", [])
            first_frame = samples[0][0] if samples else None
            rows.append({
                "label": lbl,
                "samples_count": len(samples),
                "first_seen_frame": first_frame
            })
        df = pd.DataFrame(rows).sort_values(by=["first_seen_frame", "label"])
        df.to_csv(os.path.join(save_dir, "gallery_index.csv"), index=False)

        # Record metadata
        meta = {
            "built_from_detector": self._gallery_built_from,
            "embedding_model": self._gallery_model_name,
            "total_labels": len(self.face_gallery)
        }
        pd.DataFrame([meta]).to_csv(os.path.join(save_dir, "gallery_meta.csv"), index=False)
        print(f"Face gallery saved to: {save_dir}")

    def filter_faces(self, detector=None, keep=None, remove=None):
        """
        Filter faces based on their labels in the face_gallery from the analyzer output (in-place modification).

        Parameters
        ----------
        detector: Which detector's output to filter (default is the one used to build the gallery)
        keep: Only keep these labels (list/tuple/set)
        remove: Remove these labels (list/tuple/set)
        """
        if detector is None:
            detector = self._gallery_built_from
        if detector not in self.frame_analyzer_output:
            raise ValueError(f"Detector '{detector}' analyzer is not added or has no output.")
        if not self.face_labels_by_frame:
            raise RuntimeError("No face_labels_by_frame. Did you run build_face_gallery()?")

        if keep and remove:
            raise ValueError("keep and remove cannot be used together.")

        keep_set = set(keep) if keep else None
        remove_set = set(remove) if remove else None

        outputs = self.frame_analyzer_output[detector]
        for i in range(len(outputs)):
            data = outputs[i]
            coords = data.get('coordinates', [])
            confs = data.get('confidence', [])

            labels_this_frame = self.face_labels_by_frame[i] if i < len(self.face_labels_by_frame) else []
            if not coords or not labels_this_frame:
                # 该帧无脸，或未标注
                outputs[i] = {
                    'face_count': 0,
                    'face_area': 0,
                    'confidence': [],
                    'average_confidence': 0 if 'average_confidence' in data else [],
                    'coordinates': []
                }
                continue

            # The boolean mask to decide which faces to keep
            keep_mask = []
            for lbl in labels_this_frame:
                if lbl is None:
                    keep_mask.append(False)
                elif keep_set is not None:
                    keep_mask.append(lbl in keep_set)
                elif remove_set is not None:
                    keep_mask.append(lbl not in remove_set)
                else:
                    keep_mask.append(True)  # If neither keep nor remove is specified, keep all

            # Filter coordinates and confidences
            new_coords = [c for c, k in zip(coords, keep_mask) if k]
            new_confs = [c for c, k in zip(confs, keep_mask) if k]

            # Calculate new face area and average confidence
            new_face_area = sum(int(b[2]) * int(b[3]) for b in new_coords) if new_coords else 0
            if 'average_confidence' in data:
                avg_conf = float(np.mean(new_confs)) if len(new_confs) > 0 else 0
                new_data = {
                    'face_count': len(new_coords),
                    'face_area': new_face_area,
                    'coordinates': new_coords,
                    'confidence': new_confs,
                    'average_confidence': avg_conf
                }
            else:
                new_data = {
                    'face_count': len(new_coords),
                    'face_area': new_face_area,
                    'coordinates': new_coords,
                    'confidence': new_confs
                }
            outputs[i] = new_data

        print(f"Filtering done on detector '{detector}'.")

    def show_gallery_contact_sheet(self, save_path=None, cols=8, thumb_size=(112, 112), show_window=False, window_name='Face Gallery'):
        """
        Build contact sheet of the face gallery.

        Parameters
        ----------
        save_path: If provided, save the contact sheet image to this path
        cols: Number of columns in the contact sheet
        thumb_size: Size of each thumbnail (width, height).
        show_window: If True, show the contact sheet using OpenCV window
        """
        if not self.face_gallery:
            raise RuntimeError("face_gallery is empty. Please run build_face_gallery() first.")

        thumbs = []
        labels = []
        for lbl, info in self.face_gallery.items():
            # Take the first thumbnail available
            if info.get("thumb"):
                thumbs.append(info["thumb"][0])
                labels.append(lbl)

        if not thumbs:
            print("No thumbnails available in the gallery.")
            return None

        # Uniformly resize thumbnails and add labels
        th, tw = thumb_size[1], thumb_size[0]
        norm_thumbs = []
        for img in thumbs:
            t = cv.resize(img, (tw, th))
            # Label bar
            bar_h = 18
            canvas = np.full((th + bar_h, tw, 3), 245, dtype=np.uint8)
            canvas[:th, :, :] = t
            cv.putText(canvas, labels[len(norm_thumbs)], (4, th + 14), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv.LINE_AA)
            norm_thumbs.append(canvas)

        rows = int(np.ceil(len(norm_thumbs) / float(cols)))
        cell_h, cell_w = norm_thumbs[0].shape[:2]
        sheet = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)

        for idx, img in enumerate(norm_thumbs):
            r = idx // cols
            c = idx % cols
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            sheet[y1:y2, x1:x2, :] = img

        if save_path:
            cv.imwrite(save_path, sheet)
            print(f"Contact sheet saved to: {save_path}")

        if show_window:
            cv.imshow(window_name, sheet)
            cv.waitKey(0)
            cv.destroyWindow(window_name)

        return sheet

    def find_probe_frames(self, top_n=1, log_file='probe_frames_log.txt', detector='mtcnn', method='confidence'):
        save_directory = f'{self.save_directory}/probe_frames/{detector}_{method}_probe_frames'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        self.find_probe_frames_detector = detector
        self.find_probe_frames_method = method
        frames_metric = []
        frames_confidence = []
        log_file = f'{save_directory}/{detector}_{method}_{log_file}'

        if detector in self.frame_analyzer_output:
            analyzer_output = self.frame_analyzer_output[detector]

            for i, frame_data in enumerate(analyzer_output):
                face_area = frame_data.get('face_area', 0)
                confidences = frame_data.get('confidence', [0])
                first_confidence = confidences[0] if confidences else 0
                coords = frame_data.get('coordinates', [])

                # Frontal heuristic score
                frontal_score = 0
                if coords:
                    x, y, w, h = coords[0]
                    center_x = x + w / 2
                    center_y = y + h / 2
                    aspect_ratio = w / h if h > 0 else 0

                    # Consider face to be frontal if near center and aspect ratio is natural
                    if 0.75 < aspect_ratio < 1.33 and abs(center_x - self.frame_width / 2) < 0.25 * self.frame_width:
                        frontal_score = 1  # or weight this e.g., 1.5

                # Composite metric
                metric = face_area * first_confidence * (1 + 0.5 * frontal_score)

                frames_metric.append((metric, self.frame_count[i], face_area, first_confidence, frontal_score))
                frames_confidence.append((first_confidence, self.frame_count[i]))

        else:
            print(f"{detector} analyzer is not added.")
            return []

        # Select top frames based on the specified method
        if method == 'confidence':
            top_frames = heapq.nlargest(top_n, frames_confidence, key=lambda x: x[0])
        elif method == 'metrics':
            top_frames = heapq.nlargest(top_n, frames_metric, key=lambda x: x[0])
        else:
            print(f"Unknown method: {method}. Please use 'confidence' or 'metrics'.")
            return []

        self.top_frames = top_frames

        with open(log_file, 'w') as f:
            for frame in top_frames:
                if method == 'confidence':
                    fst_conf, frame_num = frame
                    log_message = (f"Probe frame at frame number: {frame_num} with confidence score: {fst_conf} "
                                   f"by {detector}\n")
                elif method == 'metrics':
                    metric, frame_num, face_area, fst_conf, frontal_score = frame
                    log_message = (f"Probe frame at frame number: {frame_num} with metric: {metric:.2f} "
                                   f"(face_area: {face_area}, confidence score: {fst_conf:.2f}, frontal_score: {frontal_score}) "
                                   f"by detector {detector}\n")

                print(log_message.strip())
                f.write(log_message)

        return top_frames

    def print_probe_frames(self, top_frames):
        for i, frame in enumerate(top_frames):
            if len(frame) == 2:
                _, frame_number = frame
            else:
                _, frame_number, _, _ = frame
            self.print_frame(frame_number, f"Probe Frame {i+1}")

    def save_probe_frames(self, top_frames):
        save_directory = (f'{self.save_directory}/probe_frames/{self.find_probe_frames_detector}_'
                          f'{self.find_probe_frames_method}_probe_frames')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Reinitialize the video capture to ensure frames can be accessed correctly
        self.cap = cv.VideoCapture(self.video_path)

        for i, frame in enumerate(top_frames):
            if len(frame) == 2:
                _, frame_number = frame
            else:
                _, frame_number, _, _ = frame
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                save_path = os.path.join(save_directory, f'probe_frame_{i+1}.jpg')
                cv.imwrite(save_path, frame)
                print(f"Probe frame {i+1} saved at {save_path}")
            else:
                print(f"Failed to retrieve frame at frame number: {frame_number}")

        self.release_resources()

    def print_frame(self, frame_number, window_name="Frame"):
        # Reinitialize the video capture to ensure frames can be accessed correctly
        self.cap = cv.VideoCapture(self.video_path)

        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            cv.imshow(window_name, frame)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print(f"Failed to retrieve frame at frame number: {frame_number}")

    def plot_face_counts(self):
        #  Plots the number of faces against frame numbers
        for k, output in self.frame_analyzer_output.items():
            if k == "similarity":
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
            plt.plot(self.frame_count, face_counts, label=k,
                     linestyle=get_style_for_analyzer(k),
                     color=get_color_for_analyzer(k), alpha=0.75)

        plt.xlabel('Frame')
        plt.ylim(0, 5)
        plt.ylabel('Number of Faces')
        plt.title('Number of Faces vs Frame Number')
        plt.legend()
        plt.grid(True)

    def plot_face_areas(self):
        #  Plots the face area recognized by the classifiers against frame numbers
        upper_limit = 1.0
        for k, output in self.frame_analyzer_output.items():
            if k == "similarity":
                continue
            face_areas = []
            for data in output:
                if 'face_area' in data:
                    face_areas.append(data['face_area'] / self.frame_area)
                else:
                    face_areas.append(0)
            plt.plot(self.frame_count, face_areas, label=k, linestyle=get_style_for_analyzer(k),
                     color=get_color_for_analyzer(k), alpha=0.75)
            upper_limit = max(face_areas) + 0.05

        plt.xlabel('Frame')
        plt.ylim(0, upper_limit)
        plt.ylabel('Face Area Ratio')
        plt.title('Face Area Ratio vs Frame Number')
        plt.legend()
        plt.grid(True)

    def plot_average_pixel_values(self):
        #  Plot the average pixel values of the video
        plt.plot(self.frame_count, self.average_pixel_values, color=legend_colors['general'])
        plt.axhline(y=self.average_value, color=legend_colors['mean'], linestyle='--', label='Average value')
        plt.xlabel('Frame')
        plt.ylim(min(self.average_pixel_values)-5, max(self.average_pixel_values)+5)
        plt.ylabel('Average pixel value')
        plt.title('Pixel Intensity Trend across the Video')
        plt.legend()
        plt.grid(True)

    def plot_confidence_vs_frame(self):
        #  Plot the confidence as a function of frame number
        for analyzer_name, output in self.frame_analyzer_output.items():
            frame_numbers = []
            confidences = []

            for frame_num, data in enumerate(output):
                if 'confidence' in data:
                    frame_numbers.extend([frame_num] * len(data['confidence']))
                    confidences.extend(data['confidence'])

            if confidences:
                color = get_color_for_analyzer(analyzer_name)
                plt.scatter(frame_numbers, confidences, label=analyzer_name, color=color, alpha=0.75, s=10)

        plt.xlabel('Frame Number')
        plt.ylabel('Confidence')
        plt.title('Confidence vs Frame Number')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confidence(self, start_frame=0, end_frame=None):
        if end_frame is None:
            end_frame = int(self.frame_total)  # Ensure frame_total is an integer

        # Filter data for frames within the specified range
        frame_range = range(int(start_frame), min(int(end_frame), len(self.frame_count)))

        # Initialize a plot
        plt.figure(figsize=(14, 7))

        for analyzer_name, output in self.frame_analyzer_output.items():
            confidences = []
            frames = []

            for i in frame_range:
                if i < len(output):
                    frame_data = output[i]
                    if 'confidence' in frame_data and frame_data['confidence']:
                        confidences.append(frame_data['confidence'][0])  # Take the first face's confidence
                        frames.append(self.frame_count[i])
                    else:
                        confidences.append(None)  # Add None for missing confidence data
                        frames.append(self.frame_count[i])

            if confidences:
                plt.plot(frames, confidences, 'o-', label=f'{analyzer_name}_confidence_0')

        plt.xlabel('Frame Number')
        plt.ylabel('Confidence')
        plt.ylim(0.5, 1)
        plt.title('Confidence of Different Analyzers for the Face in Each Frame')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confidence_histogram(self, transparency=0.5):
        """
        Plot the confidence histogram for all analyzers with a specified transparency.
        """
        for analyzer_name, output in self.frame_analyzer_output.items():
            confidences = []
            for data in output:
                if 'confidence' in data:
                    confidences.append(data['confidence'])

            flattened_confidences = []
            for sublist in confidences:
                for item in sublist:
                    flattened_confidences.append(item)

            if flattened_confidences:
                color = get_color_for_analyzer(analyzer_name)
                plt.hist(flattened_confidences, bins=30, alpha=transparency, label=analyzer_name, color=color, edgecolor='k')

        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Histogram for All Analyzers')
        plt.legend()
        plt.grid(True)
        plt.show()

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

    def save_data_flattened(self, directory='', prefix='analyzed_flattened'):
        directory = f'{self.save_directory}/{directory}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        #  Initialize data structures for flattened data
        flattened_data = []
        #  Iterate through each frame's analyzed output
        for i, frame in enumerate(self.frame_count):
            #  Initialize a dictionary for each frame
            frame_data = {'frame': frame}

            #  Flatten and merge all analyzer data into frame_data
            for analyzer_name, results_list in self.frame_analyzer_output.items():
                result = results_list[i]  # Get the result for the current frame
                flat_result = flatten_data(result)  # Flatten the result
                flat_keys = flatten_keys(result)  # Flatten the keys

                # Pair flattened keys and values, then merge into frame_data
                for key, value in zip(flat_keys, flat_result):
                    frame_data[f"{analyzer_name}_{key}"] = value

            #  Add the average pixel value for the frame
            frame_data['avg_pixel_value'] = self.average_pixel_values[i]

            #  Append the frame data to the flattened data list
            flattened_data.append(frame_data)

        #  Create a DataFrame from the flattened data
        df = pd.DataFrame(flattened_data)

        #  Define a preferred column order
        preferred_order = ['frame', 'avg_pixel_value']
        analyzer_keys = set()

        #  Collect keys for each analyzer type, assuming they start with the analyzer's name
        for column in df.columns:
            if column.startswith(tuple(self.frame_analyzer.keys())):
                analyzer_keys.add(column.split('_')[0])  # Get the analyzer name prefix

        #  Add analyzer data to preferred order, grouped by analyzer
        for analyzer in analyzer_keys:
            preferred_order.extend([col for col in df.columns if col.startswith(analyzer)])

        #  Ensure all columns are included by adding any remaining columns at the end
        remaining_columns = [col for col in df.columns if col not in preferred_order]
        preferred_order.extend(remaining_columns)
        # preferred_order[2:] = preferred_order[2:][::-1]  # Reverse the results of analyzers

        #  Reorder the DataFrame according to the preferred order
        df = df[preferred_order]

        df.to_csv(os.path.join(directory, f'{prefix}_data.csv'), index=False)

    # ---------- Face Gallery: toolkit ----------
    @staticmethod
    def _crop_face_from_frame(frame, box):
        x, y, w, h = map(int, box)
        h_img, w_img = frame.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        return frame[y:y+h, x:x+w]

    @staticmethod
    def _l2_normalize(v):
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    @staticmethod
    def _euclidean_distance(a, b):
        return np.linalg.norm(a - b)

    def _represent_face(self, face_img_bgr, model_name='Facenet'):
        # DeepFace 期望 RGB
        face_rgb = cv.cvtColor(face_img_bgr, cv.COLOR_BGR2RGB)
        reps = DeepFace.represent(face_rgb, model_name=model_name, enforce_detection=False)
        emb = np.array(reps[0]['embedding'], dtype=np.float32)
        return self._l2_normalize(emb)

    def _update_centroid(self, old, new, count_old):
        # Update centroid with new embedding
        return self._l2_normalize((old * count_old + new) / (count_old + 1))


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


class FrameAnalyzerMTCNNIndependent:
    def __init__(self, name="mtcnn_old"):
        self.detector = MTCNN()
        self.name = name
        #  Store detected faces as well as coordinates for transfer
        self.detected_faces = []

    def analyze_frame(self, frame):
        self.detected_faces = []
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb)

        #  Faces and coordinates transfer
        frame_results = {
            'coordinates': [face['box'] for face in faces],
            'images': [frame[face['box'][1]:face['box'][1] + face['box'][3],
                             face['box'][0]:face['box'][0] + face['box'][2]]
                       for face in faces]
        }
        self.detected_faces.append(frame_results)
        confidence = self.get_confidence(faces)
        average_confidence = np.mean(confidence) if confidence else 0  # Calculate the average confidence
        face_count = len(faces)
        face_area = self.get_face_area(faces)
        coordinates = self.get_face_coordinates(faces)

        return {
            # f'{self.name}_face_count': face_count,
            f'face_count': face_count,
            f'face_area': face_area,
            f'confidence': confidence,
            f'average_confidence': average_confidence,  # Add average confidence to the output
            f'coordinates': coordinates
        }

    def get_confidence(self, faces):
        confidence = []

        for face in faces:
            confidence.append(face['confidence'])

        return confidence

    def get_face_area(self, faces):
        face_area_sum = sum(face['box'][2] * face['box'][3] for face in faces)
        return face_area_sum

    def get_face_coordinates(self, faces):
        coordinates = []

        for face in faces:
            coordinates.append(face['box'])

        return coordinates


class FrameAnalyzerOpenCVIndependent:
    def __init__(self,
                 cascade_path=str(files("pyWitnessAI.OpenCV_Models").joinpath("haarcascade_frontalface_alt.xml")),
                 name='opencv_old'):
        self.face_cascade = cv.CascadeClassifier(cascade_path)
        self.name = name

    def analyze_frame(self, frame):
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        face_count = len(faces)
        face_area = self.get_face_area(faces)
        face_coordinates = self.get_face_coordinates(faces)
        confidence = 1.0 if faces else 0
        average_confidence = np.mean(confidence) if confidence else 0

        return {
            f'face_count': face_count,
            f'face_area': face_area,
            f'coordinates': face_coordinates,
            f'confidence': confidence,
            f'average_confidence': average_confidence
        }

    def get_face_area(self, faces):
        face_area_sum = sum([w * h for (x, y, w, h) in faces])
        return face_area_sum

    def get_face_coordinates(self, faces):
        coordinates = []
        for (x, y, w, h) in faces:
            coordinates.append([x, y, w, h])
        return coordinates


class FrameAnalyzerDeepface:
    def __init__(self, detector_backend='mtcnn'):
        self.detect_backend = detector_backend
        self.name = detector_backend
        self.detected_faces = []

    def analyze_frame(self, frame):
        try:
            faces = DeepFace.extract_faces(frame, detector_backend=self.detect_backend, enforce_detection=False)
        except ValueError:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'average_confidence': 0,
                'coordinates': []
            }

        if not faces or faces is None:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'average_confidence': 0,
                'coordinates': []
            }

        valid_faces = []
        confidences = []
        coordinates = []
        face_area_sum = 0
        frame_height, frame_width = frame.shape[:2]

        for face in faces:
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], \
                         face['facial_area']['w'], face['facial_area']['h']

            # Ignore faces that cover the entire frame or have zero confidence
            if (w == frame_width and h == frame_height) or face['confidence'] == 0:
                continue

            valid_faces.append(face)
            face_area = w * h
            face_area_sum += face_area
            confidence = face['confidence']
            confidences.append(confidence)
            coordinates.append([x, y, w, h])

        if not valid_faces:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'average_confidence': 0,
                'coordinates': []
            }

        average_confidence = np.mean(confidences) if confidences else 0

        return {
            'face_count': len(valid_faces),
            'face_area': face_area_sum,
            'confidence': confidences,
            'average_confidence': average_confidence,
            'coordinates': coordinates
        }


class FrameAnalyzerMTCNN:
    def __init__(self, name='mtcnn', detector_backend='mtcnn'):
        self.detect_backend = detector_backend
        self.name = name
        self.detected_faces = []

    def analyze_frame(self, frame):
        try:
            faces = DeepFace.extract_faces(frame, detector_backend=self.detect_backend, enforce_detection=False)
        except ValueError:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        if not faces or faces is None:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        valid_faces = []
        confidences = []
        coordinates = []
        face_area_sum = 0
        frame_height, frame_width = frame.shape[:2]

        for face in faces:
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], \
                         face['facial_area']['w'], face['facial_area']['h']

            # Ignore faces that cover the entire frame or have zero confidence
            if (w == frame_width and h == frame_height) or face['confidence'] == 0:
                continue

            valid_faces.append(face)
            face_area = w * h
            face_area_sum += face_area
            confidence = face['confidence']
            confidences.append(confidence)
            coordinates.append([x, y, w, h])

        if not valid_faces:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        return {
            'face_count': len(valid_faces),
            'face_area': face_area_sum,
            'confidence': confidences,
            'coordinates': coordinates
        }


class FrameAnalyzerOpenCV:
    def __init__(self, name='opencv', detector_backend='opencv'):
        self.detect_backend = detector_backend
        self.name = name
        self.detected_faces = []

    def analyze_frame(self, frame):
        try:
            faces = DeepFace.extract_faces(frame, detector_backend=self.detect_backend, enforce_detection=False)
        except ValueError:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        if not faces or faces is None:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        valid_faces = []
        confidences = []
        coordinates = []
        face_area_sum = 0
        frame_height, frame_width = frame.shape[:2]

        for face in faces:
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], \
                         face['facial_area']['w'], face['facial_area']['h']

            # Ignore faces that cover the entire frame or have zero confidence
            if (w == frame_width and h == frame_height) or face['confidence'] == 0:
                continue

            valid_faces.append(face)
            face_area = w * h
            face_area_sum += face_area
            confidence = face['confidence']
            confidences.append(confidence)
            coordinates.append([x, y, w, h])

        if not valid_faces:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        return {
            'face_count': len(valid_faces),
            'face_area': face_area_sum,
            'confidence': confidences,
            'coordinates': coordinates
        }


class FrameAnalyzerFastMTCNN:
    def __init__(self, name='fastmtcnn', detector_backend='fastmtcnn'):
        self.detect_backend = detector_backend
        self.name = name
        self.detected_faces = []

    def analyze_frame(self, frame):
        try:
            faces = DeepFace.extract_faces(frame, detector_backend=self.detect_backend, enforce_detection=False)
        except ValueError:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        if not faces or faces is None:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        valid_faces = []
        confidences = []
        coordinates = []
        face_area_sum = 0
        frame_height, frame_width = frame.shape[:2]

        for face in faces:
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], \
                         face['facial_area']['w'], face['facial_area']['h']

            # Ignore faces that cover the entire frame or have zero confidence
            if (w == frame_width and h == frame_height) or face['confidence'] == 0:
                continue

            valid_faces.append(face)
            face_area = w * h
            face_area_sum += face_area
            confidence = face['confidence']
            confidences.append(confidence)
            coordinates.append([x, y, w, h])

        if not valid_faces:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'average_confidence': 0,
                'coordinates': []
            }

        return {
            'face_count': len(valid_faces),
            'face_area': face_area_sum,
            'confidence': confidences,
            'coordinates': coordinates
        }


class SimilarityAnalyzer:
    def __init__(self, lineup_faces, detector=None,
                 calculate_method='euclidean', model_name='Facenet', name='similarity'):
        self.name = name
        self.lineup_faces = lineup_faces
        self.calculate_method = calculate_method
        self.model_name = model_name   # The model to get the embedding

        #  Face detected from FrameAnalyzer used
        self.detector = detector

    def analyze_frame(self, frame):
        #  Use pre-detected faces for analysis
        frame_results = []

        # Access the detected faces from the current frame
        detected_faces_images = self.detector.detected_faces[0]['images']

        # for detected_face_info in detected_faces_info:
        for detected_face in detected_faces_images:
            face_comparisons = []

            # detected_face_np = self.preprocess_image(detected_face)
            embedding_results_detected = self.get_embedding(detected_face)
            emb_detected = np.array(embedding_results_detected[0]['embedding'])

            for lineup_face in self.lineup_faces:
                # lineup_face_np = self.preprocess_image(lineup_face)
                embedding_results_lineup = self.get_embedding(lineup_face)
                emb_lineup = np.array(embedding_results_lineup[0]['embedding'])

                if self.calculate_method == 'euclidean':
                    similarity_score = self.calculate_similarity_euclidean(emb_detected, emb_lineup)
                    face_comparisons.append(similarity_score)
                else:
                    raise ValueError(f"Unsupported detector backend: {self.calculate_method}")

            frame_results.append(face_comparisons)

        return {
            'facenet_distance': frame_results
        }

    def get_embedding(self, face, model_name='Facenet'):
        model_name = self.model_name

        #  Generate embedding using FaceNet
        embedding = DeepFace.represent(face, model_name=model_name, enforce_detection=False)
        return np.array(embedding)

    def calculate_similarity_euclidean(self, emb1, emb2):
        return np.linalg.norm(emb1 - emb2)

    @staticmethod
    def calculate_similarity_cosine(self, emb1, emb2):
        #  return np.linalg.norm(emb1 - emb2) #  L2 norm
        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        similarity = dot_product / (norm_emb1 * norm_emb2)
        return similarity

    def preprocess_image(self, image_np):
        #  Check if the image is in PIL format, convert to numpy array if so
        # if isinstance(image_np, Image.Image):
        #     image_np = np.array(image_np)

        #  Ensure image has 3 color channels (RGB)
        if image_np.shape[2] == 4:  # If the image has 4 x`channels (RGBA)
            image_np = image_np[:, :, :3]  # Drop the alpha channel

        #  Resize the image to 160x160 pixels using OpenCV
        resized_image = cv.resize(image_np, (160, 160))

        # Convert color from RGB to BGR
        resized_image = resized_image[:, :, ::-1]
        return resized_image


class LineupLoader:
    def __init__(self, image_paths=None, directory_path=None, target_size=(160, 160), image_number=0):
        self.directory_path = directory_path
        self.target_size = target_size
        self.lineup_images = []
        # Add or remove file types as needed
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        self.image_paths = image_paths
        self.file_names = []

        if image_paths is not None:
            self.image_paths = image_paths
            if image_number == 0:
                self.number = len(self.image_paths)
            else:
                self.number = image_number
        elif directory_path is not None:
            self.directory_path = directory_path
            # self.image_paths = self._load_paths_from_directory()
        else:
            raise ValueError("Either image paths or directory path must be provided.")

    def is_image_file(self, filename):
        return any(filename.endswith(ext) for ext in self.image_extensions)

    def preprocess_image(self, image):
        image = cv.resize(image, self.target_size)
        return image

    def load_image(self):
        count = 0
        loaded_images = []

        #  Loading images when paths are specified
        if self.image_paths is not None:
            for path in self.image_paths:
                if count >= self.number:
                    break
                image = cv.imread(path)
                processed_image = self.preprocess_image(image)
                self.lineup_images.append(processed_image)
                count += 1
            return self.lineup_images

        #  Loading images from directory
        # Loading images from directory
        if self.directory_path is not None:
            # Retrieve and sort the list of filenames
            filenames = os.listdir(self.directory_path)
            self.file_names = filenames
            # print(self.file_names)
            # Sort filenames alphabetically, case-insensitive
            sorted_filenames = sorted(self.file_names, key=lambda x: x.lower())

            for filename in sorted_filenames:
                if self.is_image_file(filename):
                    full_path = os.path.join(self.directory_path, filename)
                    image = cv.imread(full_path)
                    if image is not None:
                        processed_image = self.preprocess_image(image)
                        self.lineup_images.append(processed_image)
            return self.lineup_images

    def compare_faces(self, target_faces, filler_faces, detector='opencv', model_name='Facenet512', calculate_method='euclidean'):
        #  Use pre-detected faces for analysis
        frame_results = []
        model_name = model_name

        for target_face in target_faces:
            face_comparisons = []

            embedding_results_target = self.get_embedding(target_face, detector, model_name)
            emb_target = np.array(embedding_results_target[0]['embedding'])

            for j, filler_face in enumerate(filler_faces):
                embedding_results_filler = self.get_embedding(filler_face, detector, model_name)
                emb_filler = np.array(embedding_results_filler[0]['embedding'])

                if calculate_method == 'euclidean':
                    similarity_score = self.calculate_similarity_euclidean(emb_target, emb_filler)
                    face_comparisons.append({f'similarity_{j}': similarity_score})
                else:
                    raise ValueError(f"Unsupported detector backend: {calculate_method}")

            frame_results.append(face_comparisons)

        return frame_results

    def get_embedding(self, face, detector, model_name):
        #  Generate embedding using FaceNet
        embedding = DeepFace.represent(face, model_name=model_name, detector_backend = detector, enforce_detection=False)
        return np.array(embedding)

    def calculate_similarity_euclidean(self, emb1, emb2):
        # Apply L2 normalization to both embeddings
        emb1_normalized = emb1 / np.linalg.norm(emb1)
        emb2_normalized = emb2 / np.linalg.norm(emb2)

        # Compute the Euclidean distance between normalized embeddings
        return np.linalg.norm(emb1_normalized - emb2_normalized)

    def save(self, data, directory='results', label='', column_name=None):
        if not os.path.exists(directory):
            os.makedirs(directory)

        transposed_data = {}

        # Extract data
        for i,m in zip(range(len(data[0])), self.file_names):  # Assuming all sublists have the same length
            transposed_data[f'similarity_{m}'] = [data[j][i][f'similarity_{i}'] for j in range(len(data))]

        # Create DataFrame from the dictionary
        df = pd.DataFrame(transposed_data)

        # Transpose the DataFrame to get the desired structure
        df_transposed = df.T

        # Rename the columns to 0, 1, 2, ...
        # df_transposed.columns = [str(i) for i in range(len(data))]
        df_transposed.columns = [str(i) for i in column_name]

        # Save to CSV
        csv_filename = f"{label}_similarity_scores.csv"

        # Save to CSV
        df_transposed.to_csv(os.path.join(directory, csv_filename), index=True)
        print(f'Data saved to {csv_filename}.')

    













