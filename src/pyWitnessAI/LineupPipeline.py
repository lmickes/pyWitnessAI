import cv2 as cv
from typing import List, Optional, Literal
from dataclasses import dataclass

from .ImageAnalyzer import *
from .LineupLoader import LineupLoader
from .LineupIdentifier import LineupIdentifier


@dataclass
class PipelineConfig:
    analyzer_method: Literal["process", "process_verify", "process_georgia_pipeline"] = "process"
    distance_metric: str = "euclidean_l2"
    backend: str = "opencv"
    model: str = "VGG-Face"
    align: bool = False
    normalization: str = "base"
    enforce_detection: bool = False
    show_progress: bool = False
    # Use full frame or mtcnn to detect faces in each frame of the video
    detect_faces: Literal["fullframe", "mtcnn"] = "fullframe"
    # Fallback distance when no face detected in a frame, preferably with detect_faces as "mtcnn"
    no_face_fill: float = 50.0


def export_for_pywitness(
    input_csv: str,
    output_csv: Optional[str] = None,
    targetLineup: Literal["targetPresent", "targetAbsent", "both"] = "both",
    lineupSize: int = 6,
) -> str:
    """
    Convert a long-form pipeline CSV into a minimal pyWitness-ready CSV.

    Input expectations: Columns include the per-frame summary fields (TA_responseType, TP_responseType, TA_conf, TP_conf)

    Parameters
    ------
    input_csv: Path to the pipeline CSV file.
    output_csv: Where to write the new CSV. If None, will write next to input as '*_pywitness.csv'.
    targetLineup : {'targetPresent','targetAbsent','both'}
        - 'targetPresent'   -> use TP_responseType / TP_conf
        - 'targetAbsent'    -> use TA_responseType / TA_conf
        - 'both'            -> split each input record into TWO output rows (TP and TA)
    lineupSize: Number written into the `lineupSize` column for every output row.

    Output schema:
        responseType, confidence, targetLineup, lineupSize
    """
    df = pd.read_csv(input_csv)

    if targetLineup is None:
        targetLineup = "both"

    # Sanity checks
    need_cols = {"TA_responseType","TP_responseType","TA_conf","TP_conf"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    has_guilty = False
    has_innocent = False

    # Check if guilty_suspect and innocent_suspect exist in the data
    cols = [str(c) for c in df.columns]
    dist_cols = [c for c in cols if c.startswith("dist[")]
    if dist_cols:
        has_guilty   = any("[guilty_suspect]"   in c for c in dist_cols)
        has_innocent = any("[innocent_suspect]" in c for c in dist_cols)

    # When data is not wide-form, check "role" column
    if ("role" in df.columns) and (not dist_cols):
        unique_roles = set(df["role"].dropna().unique().tolist())
        has_guilty   = has_guilty   or ("guilty_suspect"   in unique_roles)
        has_innocent = has_innocent or ("innocent_suspect" in unique_roles)

    effective_lineup_size_both = int(lineupSize) - 1 if (targetLineup == "both") else int(lineupSize)
    # Avoid non-positive lineup size
    if effective_lineup_size_both < 1:
        effective_lineup_size_both = 1

    # We want one output row per group (frame, probe)
    group_keys = [c for c in ["frame", "probe"] if c in df.columns]
    if group_keys:
        summary = (
            df.groupby(group_keys, as_index=False)[["TA_responseType","TP_responseType","TA_conf","TP_conf"]]
              .first()
        )
    else:
        # If neither frame nor probe exists, just take the first row
        summary = df[["TA_responseType","TP_responseType","TA_conf","TP_conf"]].drop_duplicates().head(1)
        summary["__dummy__"] = 0
        group_keys = ["__dummy__"]

    out_rows = []

    def emit_row(resp: str, conf, tl: str, lineup_size=None):
        out_rows.append({
            "responseType": resp if pd.notna(resp) else None,
            "confidence":   float(conf) if pd.notna(conf) else None,
            "targetLineup": tl,
            "lineupSize":   int(lineup_size if lineup_size is not None else lineupSize),
        })

    # For each summary row, emit one or two output rows depending on targetLineup
    for _, row in summary.iterrows():
        ta_resp, ta_conf = row.get("TA_responseType"), row.get("TA_conf")
        tp_resp, tp_conf = row.get("TP_responseType"), row.get("TP_conf")

        if targetLineup == "targetPresent":
            emit_row(tp_resp, tp_conf, "targetPresent")
        elif targetLineup == "targetAbsent":
            emit_row(ta_resp, ta_conf, "targetAbsent")
        elif targetLineup == "both":
            emit_row(tp_resp, tp_conf, "targetPresent", effective_lineup_size_both)
            emit_row(ta_resp, ta_conf, "targetAbsent", effective_lineup_size_both)
        else:
            raise ValueError("targetLineup must be 'targetPresent', 'targetAbsent', or 'both'.")

    out = pd.DataFrame(out_rows, columns=["responseType","confidence","targetLineup","lineupSize"])

    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_pywitness.csv"

    out.to_csv(output_csv, index=False)
    return output_csv


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
                 cfg: PipelineConfig | dict = PipelineConfig(),
                 identifier: Optional[LineupIdentifier] = None):
        self.video_path = video_path
        self.lineup_loader = lineup_loader

        if isinstance(cfg, dict):
            self.cfg = PipelineConfig(**cfg)
        elif isinstance(cfg, PipelineConfig):
            self.cfg = cfg
        else:
            raise TypeError("cfg must be a PipelineConfig or dict")

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

    def _has_face(self, pil_img: Image.Image) -> bool:
        """
        Lightweight face presence check using the configured backend.
        - If backend == 'mtcnn': reuse analyzer's MTCNN (fast, no extra deps).
        - Else: use DeepFace.extract_faces with the chosen backend.
        Returns True if at least one face is detected, else False.
        """
        backend = getattr(self.cfg, "backend", "mtcnn")

        # Fast path for MTCNN
        if str(backend).lower() == "mtcnn":
            try:
                boxes, probs = self._analyzer_template.mtcnn.detect(pil_img)
                if boxes is None or probs is None:
                    return False
                w, h = pil_img.size
                img_area = float(w * h)
                min_area = 0.01 * img_area
                max_area = 0.80 * img_area
                for (x1, y1, x2, y2), p in zip(boxes, probs):
                    if p is None:
                        continue
                    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                    if p >= 0.90 and (min_area <= area <= max_area):  # 置信度/面积双阈值
                        return True
                return False
            except Exception:
                return False

        # Generic path for other backends supported by DeepFace
        try:
            arr = np.array(pil_img)
            faces = DeepFace.extract_faces(
                img_path=arr,
                detector_backend=backend,
                enforce_detection=False
            )
            for f in faces or []:
                conf = float(f.get("confidence", 0.0))
                region = f.get("facial_area") or f.get("region") or {}
                w, h = pil_img.size
                img_area = float(w * h)
                if isinstance(region, dict):
                    rw = float(region.get("w", 0.0))
                    rh = float(region.get("h", 0.0))
                    area = rw * rh
                else:
                    area = 0.0
                if conf >= 0.90 and area >= 0.01 * img_area and area <= 0.80 * img_area:
                    return True
            return False
        except Exception:
            return False

    def _faces_from_frame(self, frame_bgr: np.ndarray) -> List[Image.Image]:
        """
        Return a list of PIL faces for this frame.
        - "fullframe": treat whole frame as a single face (current default behaviour)
        - "mtcnn": use facenet-pytorch MTCNN; if none found, return []
        """
        img_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        if self.cfg.detect_faces == "fullframe":
            if self._has_face(pil):
                return [pil]
            else:
                return []

        # mtcnn path (use analyzer's MTCNN to avoid new deps)
        face = self._analyzer_template.mtcnn(pil)  # torch.Tensor [3,H,W] in [0,1], or None
        if face is None:
            return []
        face_np = (face.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
        return [Image.fromarray(face_np)]

    def _normalize_summary(self, summary: Dict[str, object]) -> Dict[str, object]:
        """
        Ensure all expected keys are present in the summary dict.
        """
        base = {
            "filler_min": None,
            "selected": None,
            "responseType": None,
            "confidence": None,
            "TA_selected": None,
            "TP_selected": None,
            "TA_responseType": None,
            "TP_responseType": None,
            "TA_conf": None,
            "TP_conf": None,
        }
        base.update(summary or {})
        return base

    def _decide_summary(self, sim_df: pd.DataFrame) -> Dict[str, object]:
        """
        Decide the summary for the frame based on the identifier settings and roles.
        Choices: decide_macro, decide_tp_or_ta, decide (simple)
        """
        if not self.identifier_enabled or self.identifier_obj is None:
            return {}

        roles = self.roles
        has_guilty = any(v == "guilty_suspect" for v in roles.values())
        has_innocent = any(v == "innocent_suspect" for v in roles.values())

        # Case1: no targetLineup, both guilty and innocent exist -> macro
        if self.identifier_obj.targetLineup is None and has_guilty and has_innocent:
            return self._normalize_summary(self.identifier_obj.decide_macro(sim_df, roles))

        # Case2: targetLineup is set to TP/TA -> TP/TA strategy
        if self.identifier_obj.targetLineup in ("targetPresent", "targetAbsent"):
            return self._normalize_summary(self.identifier_obj.decide_tp_or_ta(sim_df, roles))

        # Case3: neither guilty nor innocent exists, or only one exists but targetLineup is not set
        return self._normalize_summary(self.identifier_obj.decide_for_pipeline(sim_df, self.lineup_loader))

    def _long_to_wide(self, df: pd.DataFrame, by=("frame", "probe"),
                      include_role_mins=True,
                      name_by_role=True,  # True: dist[member[role]]; False: dist[member]
    ) -> pd.DataFrame:
        """
        # Convert long-form df to wide-form for intuitionistic and easier analysis.
        # Columns to keep as index (frame, probe) or just (frame)
        # include_role_mins: whether to include min distances by role as separate columns
        # name_by_role: whether to include role in the distance column names
        """
        dist_wide = (
            df.pivot_table(index=list(by), columns="lineup_member", values="distance", aggfunc="min")
            .sort_index(axis=1)
        )

        # The pattern dist[member[role]] or dist[member]
        if name_by_role:
            # Get role map from df (not self.roles, in case some members missing)
            role_map = (
                df[["lineup_member", "role"]]
                .dropna()
                .drop_duplicates(subset=["lineup_member"], keep="first")
                .set_index("lineup_member")["role"]
                .to_dict()
            )
            dist_wide.columns = [f"{member}[{role_map.get(member, 'unknown')}]" for member in dist_wide.columns]
        else:
            dist_wide.columns = [f"dist[{c}]" for c in dist_wide.columns]

        # Summary columns
        candidate_summary_cols = [
            "filler_min", "TA_selected", "TP_selected", "TA_responseType", "TP_responseType", "TA_conf",
            "TP_conf" , "selected", "responseType", "confidence",
        ]
        summary_cols = [c for c in candidate_summary_cols if c in df.columns]
        summary = df.groupby(list(by), as_index=True)[summary_cols].first()

        # Optionally include min distances by role
        if include_role_mins and "role" in df.columns:
            role_min = df.groupby(list(by) + ["role"])["distance"].min().unstack("role")
            role_min = role_min.add_prefix("min_by_role[")
            role_min.columns = [c + "]" for c in role_min.columns]
            out = pd.concat([summary, role_min, dist_wide], axis=1).reset_index()
        else:
            out = pd.concat([summary, dist_wide], axis=1).reset_index()

        return out

    def run(self,
            output_csv: Optional[str] = None,
            wide: bool = True,
            export_pywitness: bool = False,
            frame_start: int = 0,
            frame_end: Optional[int] = None,
            frame_stride: int = 1):
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 0
        if n_frames <= 0:
            n_frames = 10 ** 12

        # Support negative indexing for frame_start and frame_end
        if frame_start < 0:
            frame_start = max(0, n_frames + frame_start)
        if frame_end is None:
            frame_end = n_frames - 1
        elif frame_end < 0:
            frame_end = max(-1, n_frames + frame_end)

        frame_start = max(0, frame_start)
        frame_end = max(-1, frame_end)
        if frame_end >= 0:
            frame_end = min(frame_end, n_frames - 1)

        if frame_start > frame_end and frame_end >= 0:
            # If start > end (and end is valid), no frames to process
            return pd.DataFrame()

        frame_stride = max(1, int(frame_stride))

        # Set the video to start frame
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_start)

        records = []
        logical_frame_idx = frame_start - 1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            logical_frame_idx += 1

            if logical_frame_idx > frame_end:
                break

            if (logical_frame_idx - frame_start) % frame_stride != 0:
                continue

            faces = self._faces_from_frame(frame)  # List[PIL.Image]
            print(f"[DEBUG] frame={logical_frame_idx}, detected_faces={len(faces)}, mode={self.cfg.detect_faces}")

            if len(faces) == 0:
                # Avoid None as index when it is passed to pivot_table
                noface_probe = f"frame{logical_frame_idx}_noface"
                for lm_path in self.lineup_loader.lineup:
                    lm_name = lm_path.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
                    role = self.roles.get(lm_name, "filler")

                    base_row = {
                        "frame": logical_frame_idx,
                        "probe": noface_probe,
                        "lineup_member": lm_name,
                        "distance": float(self.cfg.no_face_fill),  # Fill fixed value when no face detected
                        "role": role,
                    }

                    if self.identifier_enabled:
                        summary = self._normalize_summary({
                            "filler_min": float(self.cfg.no_face_fill),
                            "selected": None,
                            "responseType": "fillerId",
                            "confidence": float(self.cfg.no_face_fill),
                        })
                        records.append({**base_row, **summary})
                    else:
                        records.append(base_row)
                continue

            # Build column images for the frame
            probe_names = [f"frame{logical_frame_idx}_face{i}" for i in range(len(faces))]
            col_il = ImageLoader([])
            col_il.images = {n: im for n, im in zip(probe_names, faces)}
            col_il.path_to_images = {n: f"{n}.png" for n in probe_names}  # Virtual paths

            analyzer = self._analyzer_template
            analyzer.column_images = col_il

            # Gain the updated progress bar
            analyzer.show_progress = self.cfg.show_progress

            getattr(analyzer, self.cfg.analyzer_method)()
            sim_df = analyzer.dataframe()
            sim_df = sim_df.T  # Rows: probe, Columns: lineup member (fits LineupIdentifier expectations)

            # If Identifier is set, get summary for the frame
            summary = {}
            if self.identifier_enabled:
                summary = self._decide_summary(sim_df)

            # Expand sim_df to long-form records
            # Each row: (frame, probe, lineup_member, distance, role, **summary
            # per_member_min = sim_df.min(axis=1)

            for probe, row in sim_df.iterrows():
                for lm_name, dist in row.items():
                    role = self.roles.get(lm_name, "filler")
                    base_row = {
                        "frame": logical_frame_idx,
                        "probe": probe,
                        "lineup_member": lm_name,
                        "distance": (None if pd.isna(dist) else float(dist)),
                        "role": role,
                    }
                    if self.identifier_enabled:
                        records.append({**base_row, **summary})
                    else:
                        records.append(base_row)

        cap.release()
        df = pd.DataFrame.from_records(records)
        if wide:
            df = self._long_to_wide(df, by=("frame", "probe"))
        if output_csv:
            df.to_csv(output_csv, index=False)
        if export_pywitness:
            df.to_csv(output_csv, index=False)
            checked_target = (
                self.identifier_obj.targetLineup
                if (self.identifier_obj and self.identifier_obj.targetLineup in ("targetPresent", "targetAbsent"))
                else "both"
            )
            export_for_pywitness(
                input_csv=output_csv,
                output_csv=output_csv.replace(".csv", "_pywitness.csv"),
                targetLineup=checked_target,
                lineupSize=len(self.lineup_loader.lineup),
            )
        return df
