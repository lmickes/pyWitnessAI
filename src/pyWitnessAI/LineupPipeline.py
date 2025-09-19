import cv2 as cv
from typing import List, Optional, Literal
from dataclasses import dataclass

from .ImageAnalyzer import *
from .LineupLoader import LineupLoader
from .LineupIdentifier import LineupIdentifier


@dataclass
class PipelineConfig:
    analyzer_method: Literal["process", "process_verify", "process_georgia_pipeline"] = "process"
    distance_metric: str = "euclidean"
    backend: str = "opencv"
    model: str = "VGG-Face"
    align: bool = False
    normalization: str = "base"
    enforce_detection: bool = False
    show_progress: bool = False

class VideoLineupPipeline:
    """
    For each frame:
      - Row data: frame + lineup member distances + TA/TP summary (if identifier is set)
      - Distance is calculated by ImageAnalyzer between detected faces and lineup members
      - Summary is calculated by LineupIdentifier from the distance matrix
    """

    def __init__(self,
                 video_path: str,
                 lineup_loader: LineupLoader,
                 cfg: PipelineConfig = PipelineConfig(),
                 identifier: Optional[LineupIdentifier] = None):
        self.video_path = video_path
        self.lineup_loader = lineup_loader
        self.cfg = cfg

        # Normalize identifier switch/object
        if isinstance(identifier, LineupIdentifier):
            self.identifier_obj: Optional[LineupIdentifier] = identifier
            self.identifier_enabled: bool = True
        elif identifier is True:
            self.identifier_obj = LineupIdentifier()  # default threshold/targetLineup/target
            self.identifier_enabled = True
        else:
            self.identifier_obj = None
            self.identifier_enabled = False

        # Initialize the lineup rows from LineupLoader
        self._row_il = ImageLoader([*self.lineup_loader.lineup])  # paths -> PIL
        # Roles map: image base name -> role
        self.roles = LineupIdentifier._role_map_from_lineuploader(self.lineup_loader)

        self._analyzer_template = ImageAnalyzer(
            column_images=ImageLoader([]),
            row_images=self._row_il,
            distance_metric=self.cfg.distance_metric,
            backend=self.cfg.backend,
            enforce_detection=self.cfg.enforce_detection,
            model=self.cfg.model,
            align=self.cfg.align,
            normalization=self.cfg.normalization,
            show_progress=self.cfg.show_progress
        )

    def _faces_from_frame(self, frame_bgr: np.ndarray) -> List[Image.Image]:
        """
        Mtcnn can also be used here for better face detection/cropping.
        """
        # Treat the whole frame as one face for simplicity
        # Return pil for the consistency with ImageLoader
        img_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        return [pil]  # !!!!! can be multiple faces in future?

    def run(self, output_csv: str) -> str:
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        records = []
        frame_idx = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            faces = self._faces_from_frame(frame)  # List[PIL.Image]
            if len(faces) == 0:
                # If no faces detected, record fillers with distance=50.0
                for lm_path in self.lineup_loader.lineup:
                    lm_name = lm_path.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
                    role = self.roles.get(lm_name, "filler")
                    records.append({
                        "frame": frame_idx,
                        "probe": None,
                        "lineup_member": lm_name,
                        "distance": 50.0,
                        "role": role,
                        # Summary columns for no-detection case
                        "filler_min": 50.0,
                        "selected": None,
                        "responseType": "fillerId",
                        "confidence": 50.0,
                        # Macro columns
                        "TA_selected": None, "TP_selected": None,
                        "TA_responseType": None, "TP_responseType": None,
                        "TA_conf": None, "TP_conf": None,
                    })
                continue

            # Build column images for the frame
            probe_names = [f"frame{frame_idx}_face{i}" for i in range(len(faces))]
            col_il = ImageLoader([])
            col_il.images = {n: im for n, im in zip(probe_names, faces)}
            col_il.path_to_images = {n: f"{n}.png" for n in probe_names}  # Virtual paths

            analyzer = self._analyzer_template
            analyzer.column_images = col_il

            # Gain the updated progress bar
            analyzer.show_progress = self.cfg.show_progress

            getattr(analyzer, self.cfg.analyzer_method)()
            sim_df = analyzer.dataframe()
            sim_df = sim_df.T  # Rows: probe, Columns: lineup member

            # If Identifier is set, get summary for the frame
            summary = {
                # None if no faces detected
                "filler_min": None, "selected": None, "responseType": None, "confidence": None,
                "TA_selected": None, "TP_selected": None,
                "TA_responseType": None, "TP_responseType": None,
                "TA_conf": None, "TP_conf": None,
            }
            if self.identifier is not None:
                summ = self.identifier.decide(sim_df, self.lineup_loader)
                summary.update(summ)

            # Expand sim_df to long-form records
            # Each row: (frame, probe, lineup_member, distance, role, **summary
            per_member_min = sim_df.min(axis=1)

            for probe, row in sim_df.iterrows():
                for lm_name, dist in row.items():  # each lineup member
                    role = self.roles.get(lm_name, "filler")
                    records.append({
                        "frame": frame_idx,
                        "probe": probe,
                        "lineup_member": lm_name,
                        "distance": (None if pd.isna(dist) else float(dist)),
                        "role": role,
                        **summary
                    })
        cap.release()
        df = pd.DataFrame.from_records(records)

        df.to_csv(output_csv, index=False)
        return output_csv
