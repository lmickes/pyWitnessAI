import cv2
from tqdm import tqdm
from .ImageAnalyzer import *
from .LineupDecider import *
from facenet_pytorch import MTCNN
import math

class VideoLineupPipeline:
    """
    VideoLineupPipeline: Make identification decisions on a video using a lineup of images.
    This pipeline processes a video frame by frame, computes embeddings for each frame,
    and compares them against a lineup of images to make identification decisions.
    """
    def __init__(
        self,
        lineup_glob: list[str] | str,
        suspect_path: Optional[str],
        pipeline: Literal["georgia", "deepface", "deepface_verify"] = "georgia",  # 新增 deepface_verify
        lineup_backend: str = "opencv",
        deepface_model: str = "VGG-Face",
        normalization: str = "base",
        enforce_detection: bool = False,
        show_progress: bool = True,
        device: Optional[str] = None,
        verify_distance_metric: str = "cosine",      # DeepFace.verify - distance_metric
    ):
        # 载入阵容图像（行）
        if isinstance(lineup_glob, str):
            lineup_patterns = [lineup_glob]
        else:
            lineup_patterns = list(lineup_glob)

        self.lineup_images = ImageLoader(lineup_patterns)
        # If suspect_path is provided, add it to the lineup
        self.suspect_idx = None
        if suspect_path is not None:
            sus_img = Image.open(suspect_path).convert("RGB")
            sus_name = os.path.splitext(os.path.basename(suspect_path))[0]
            self.lineup_images.images[sus_name] = sus_img
            self.lineup_images.path_to_images[sus_name] = suspect_path
            self.suspect_idx = sus_name

        # Analyzer initialization
        # Give column_images an empty ImageLoader, as it is not used in this pipeline
        self.pipeline = pipeline
        self.show_progress = show_progress
        self.device = device
        self.verify_distance_metric = verify_distance_metric   # 存起来

        self.analyzer = ImageAnalyzer(
            column_images=ImageLoader([]),
            row_images=self.lineup_images,
            backend=lineup_backend,
            model=deepface_model,
            normalization=normalization,
            enforce_detection=enforce_detection,
            show_progress=False,
            device=device
        )
        self.mtcnn_all = MTCNN(keep_all=True, device=self.analyzer.device)

        # pipeline-specific pre-processing
        if self.pipeline == "georgia":
            self._lineup_embs = {}
            for k, img in tqdm(self.lineup_images.images.items(), desc="Precompute lineup (FaceNet)", disable=not self.show_progress):
                self._lineup_embs[k] = self.analyzer.get_embedding_facenet(img)

        elif self.pipeline == "deepface":
            self._lineup_embs = {}
            for k, img in tqdm(self.lineup_images.images.items(), desc="Precompute lineup (DeepFace)", disable=not self.show_progress):
                try:
                    self._lineup_embs[k] = self.analyzer.get_embedding(img)
                except Exception:
                    self._lineup_embs[k] = None

        elif self.pipeline == "deepface_verify":
            # DeepFace.verify
            self._lineup_np_bgr = {}
            for k, img in tqdm(self.lineup_images.images.items(), desc="Prepare lineup (BGR for verify)", disable=not self.show_progress):
                rgb = np.array(img)  # PIL RGB -> ndarray RGB
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self._lineup_np_bgr[k] = bgr

        else:
            raise ValueError("pipeline must be 'georgia', 'deepface', or 'deepface_verify'.")

    # Calculate embedding for a single frame
    def _frame_embedding(self, frame_rgb: Image.Image):
        if self.pipeline == "georgia":
            return self.analyzer.get_embedding_facenet(frame_rgb)  # Tensor 或 None
        else:  # deepface
            try:
                return self.analyzer.get_embedding(frame_rgb)  # ndarray
            except Exception:
                return None

    # Transform input to numpy
    @staticmethod
    def _to_numpy(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    @staticmethod
    def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _distances_to_lineup(self, frame_emb) -> Dict[str, float]:
        """
        Calculate distances from the frame embedding to each lineup member's embedding.
        """
        f = self._to_numpy(frame_emb)
        d: Dict[str, float] = {}
        for k, e in self._lineup_embs.items():
            e_np = self._to_numpy(e)
            if f is None or e_np is None:
                d[k] = np.nan
            else:
                d[k] = self._euclidean(f, e_np)
        return d

    # DeepFace verify distances
    def _distances_deepface_verify(self, frame_bgr) -> dict[str, float]:
        d = {}
        for k, ref_bgr in self._lineup_np_bgr.items():
            try:
                result = DeepFace.verify(
                    frame_bgr, ref_bgr,
                    enforce_detection=self.analyzer.enforce_detection,
                    model_name=self.analyzer.model,
                    detector_backend=self.analyzer.backend,
                    distance_metric=self.verify_distance_metric,
                    align=self.analyzer.align,
                    normalization=self.analyzer.normalization
                )
                d[k] = float(result["distance"])
            except Exception:
                d[k] = np.nan
        return d

    def run(
        self,
        video_path: str,
        decider: LineupDecider,
        output_csv: str = "video_lineup_results.csv",
        frame_stride: int = 1,
        max_frames: Optional[int] = None
    ) -> str:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        rows = []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
        to_process = total if (total is not None and max_frames is None) else (max_frames or total)
        pbar_total = to_process // frame_stride if (to_process and frame_stride > 1) else to_process
        pbar = tqdm(total=pbar_total, desc="Frame recognition", disable=not self.show_progress) if pbar_total else None

        frame_idx, processed = 0, 0
        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                if frame_idx % frame_stride != 0:
                    frame_idx += 1
                    continue

                # Use MTCNN to detect faces, so that we can set 50.0 for no face detected
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                face_tensor = self.analyzer.mtcnn(pil_img)
                face_detected = face_tensor is not None

                if self.pipeline == "deepface_verify":
                    if face_detected:
                        dist_map = self._distances_deepface_verify(bgr)
                    else:
                        dist_map = {k: 50.0 for k in self.lineup_images.images.keys()}

                elif self.pipeline in ("georgia", "deepface"):
                    # georgia/deepface pipeline: Calculate embedding and distances
                    if face_detected:
                        frame_emb = self._frame_embedding(pil_img)
                        dist_map = self._distances_to_lineup(frame_emb)
                    else:
                        dist_map = {k: 50.0 for k in self.lineup_images.images.keys()}
                else:
                    raise RuntimeError("Unknown pipeline")

                # Single column DataFrame for decision-making
                col_name = decider.column_name or f"frame_{frame_idx}"
                sim_df = pd.DataFrame({col_name: pd.Series(dist_map)})

                # Check if there is a face detected and set decision if none
                if face_detected:
                    decision = decider.decide(sim_df)
                else:
                    decision = LineupDecisionResult(
                        responseType="fillerId",
                        confidence=50.0,
                        selected_id=None,
                        selected_rank=None,
                        threshold=decider.threshold,
                        targetLineup=decider.targetLineup,
                        suspect=decider.suspect,
                        column_used=col_name,
                        extras={"reason": "no_face_detected"}
                    )

                row = {
                    "frame_index": frame_idx,
                    "time_sec": (frame_idx / fps) if fps > 0 else np.nan,
                    "face_detected": bool(face_detected),
                    "responseType": decision.responseType,
                    "confidence": decision.confidence,
                    "selected_id": decision.selected_id,
                    "selected_rank": decision.selected_rank,
                    "threshold": decision.threshold,
                    "targetLineup": decision.targetLineup,
                    "suspect": decision.suspect,
                }
                for k, v in dist_map.items():
                    row[f"score::{k}"] = v
                rows.append(row)

                frame_idx += 1
                processed += 1
                if pbar: pbar.update(1)
                if max_frames is not None and processed >= max_frames:
                    break
        finally:
            if pbar: pbar.close()
            cap.release()

        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df_out = pd.DataFrame(rows)
        df_out.to_csv(output_csv, index=False)
        return output_csv

    @staticmethod
    def _crop_face(bgr_frame, box, pad=0):
        h, w = bgr_frame.shape[:2]
        x1, y1, x2, y2 = box
        x1 = int(max(0, math.floor(x1 - pad)))
        y1 = int(max(0, math.floor(y1 - pad)))
        x2 = int(min(w, math.ceil(x2 + pad)))
        y2 = int(min(h, math.ceil(y2 + pad)))
        if x2 <= x1 or y2 <= y1:
            return None
        return bgr_frame[y1:y2, x1:x2].copy()

    def _distances_deepface_verify_face(self, face_bgr) -> dict[str, float]:
        from deepface import DeepFace
        import numpy as np
        d = {}
        for k, ref_bgr in self._lineup_np_bgr.items():
            try:
                res = DeepFace.verify(
                    face_bgr, ref_bgr,
                    enforce_detection=False,  # 关键：已是人脸裁剪，无需强制再次检测
                    model_name=self.analyzer.model,
                    detector_backend=self.analyzer.backend,  # 若仍想启用对齐，可保留
                    distance_metric=self.verify_distance_metric,
                    align=self.analyzer.align,
                    normalization=self.analyzer.normalization
                )
                d[k] = float(res["distance"])
            except Exception:
                d[k] = np.nan
        return d

    def _distances_to_lineup_face(self, face_pil) -> dict[str, float]:
        # georgia: MTCNN+InceptionResnetV1；deepface: DeepFace.represent
        face_emb = self._frame_embedding(face_pil)  # Can be ndarray, tensor or None
        return self._distances_to_lineup(face_emb)

    # When there are multiple faces in a frame, we need to process each face separately
    def run_multi(
            self,
            video_path: str,
            decider,  # Can be LineupDecider or VerifyStyleDecider
            output_csv: str = "video_lineup_results_multi.csv",
            frame_stride: int = 1,
            max_frames: Optional[int] = None,
            min_prob: float = 0.90,  # Threshold for face detection confidence
            pad: int = 0,  # The padding to apply around detected faces
            output_frame_summary_csv: Optional[str] = "video_lineup_results_multi_frame.csv",  # Summary of each frame
    ) -> str:

        import cv2, numpy as np
        from PIL import Image
        from tqdm import tqdm
        import math

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        face_rows = []  # Individual face rows
        frame_summary_rows = []  # Summary rows for each frame

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
        to_process = total if (total is not None and max_frames is None) else (max_frames or total)
        pbar_total = to_process // frame_stride if (to_process and frame_stride > 1) else to_process
        pbar = tqdm(total=pbar_total, desc="Frame recognition (multi-face)",
                    disable=not self.show_progress) if pbar_total else None

        frame_idx, processed = 0, 0
        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                if frame_idx % frame_stride != 0:
                    frame_idx += 1
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                # Multi-face detection
                boxes, probs = self.mtcnn_all.detect(pil_img)
                has_faces = boxes is not None and len(boxes) > 0

                # Create a DataFrame for this frame's distances
                per_frame_candidates = []  # dict(decision=..., dist_map=..., i=..., box=...)

                if not has_faces:
                    # No faces detected: 50.0 for all distances
                    dist_map = {k: 50.0 for k in self.lineup_images.images.keys()}
                    col_name = getattr(decider, "column_name", None) or f"frame_{frame_idx}_face_none"
                    sim_df = pd.DataFrame({col_name: pd.Series(dist_map)})

                    face_row = {
                        "frame_index": frame_idx,
                        "time_sec": (frame_idx / fps) if fps > 0 else np.nan,
                        "face_detected": False,
                        "face_index": None,
                        "bbox_x1": None, "bbox_y1": None, "bbox_x2": None, "bbox_y2": None,
                        "bbox_w": None, "bbox_h": None,
                        "responseType": "fillerId",
                        "confidence": 50.0,
                        "selected_id": None,
                        "selected_rank": None,
                        "threshold": getattr(decider, "threshold", np.nan),
                        "targetLineup": getattr(decider, "targetLineup", "unknown"),
                        "suspect": getattr(decider, "suspect", None),
                    }
                    for k, v in dist_map.items():
                        face_row[f"score::{k}"] = v
                    face_rows.append(face_row)

                    # Single frame summary row
                    frame_summary_rows.append({
                        "frame_index": frame_idx,
                        "time_sec": (frame_idx / fps) if fps > 0 else np.nan,
                        "has_faces": False,
                        "chosen_face_index": None,
                        "bbox_x1": None, "bbox_y1": None, "bbox_x2": None, "bbox_y2": None,
                        "bbox_w": None, "bbox_h": None,
                        "responseType": "fillerId",
                        "confidence": 50.0,
                        "selected_id": None,
                        "selected_rank": None,
                        "threshold": getattr(decider, "threshold", np.nan),
                        "targetLineup": getattr(decider, "targetLineup", "unknown"),
                        "suspect": getattr(decider, "suspect", None),
                    })

                    frame_idx += 1
                    processed += 1
                    if pbar: pbar.update(1)
                    if max_frames is not None and processed >= max_frames:
                        break
                    continue

                # Process each detected face
                for i, (box, p) in enumerate(zip(boxes, probs)):
                    if p is None or p < min_prob:
                        continue

                    face_bgr = self._crop_face(bgr, box, pad=pad)
                    if face_bgr is None or face_bgr.size == 0:
                        continue

                    # Calculate distances for this face
                    if self.pipeline == "deepface_verify":
                        dist_map = self._distances_deepface_verify_face(face_bgr)
                    elif self.pipeline in ("georgia", "deepface"):
                        face_pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
                        dist_map = self._distances_to_lineup_face(face_pil)
                    else:
                        raise RuntimeError("Unknown pipeline")

                    # Make decision
                    col_name = getattr(decider, "column_name", None) or f"frame_{frame_idx}_face_{i}"
                    sim_df = pd.DataFrame({col_name: pd.Series(dist_map)})
                    decision = decider.decide(sim_df)

                    x1, y1, x2, y2 = [float(x) for x in box]
                    w, h = float(x2 - x1), float(y2 - y1)

                    # per-face row
                    face_row = {
                        "frame_index": frame_idx,
                        "time_sec": (frame_idx / fps) if fps > 0 else np.nan,
                        "face_detected": True,
                        "face_index": int(i),
                        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                        "bbox_w": w, "bbox_h": h,
                        "responseType": decision.responseType,
                        "confidence": decision.confidence,
                        "selected_id": decision.selected_id,
                        "selected_rank": decision.selected_rank,
                        "threshold": decision.threshold if hasattr(decider, "threshold") else np.nan,
                        "targetLineup": decision.targetLineup,
                        "suspect": decision.suspect,
                    }
                    for k, v in dist_map.items():
                        face_row[f"score::{k}"] = v
                    face_rows.append(face_row)

                    # Collect per-frame candidates for summary
                    per_frame_candidates.append({
                        "decision": decision,
                        "dist_map": dist_map,
                        "i": int(i),
                        "box": (x1, y1, x2, y2, w, h),
                    })

                # Summary for this frame: select the best candidate
                if len(per_frame_candidates) == 0:
                    # No valid faces detected
                    frame_summary_rows.append({
                        "frame_index": frame_idx,
                        "time_sec": (frame_idx / fps) if fps > 0 else np.nan,
                        "has_faces": False,
                        "chosen_face_index": None,
                        "bbox_x1": None, "bbox_y1": None, "bbox_x2": None, "bbox_y2": None,
                        "bbox_w": None, "bbox_h": None,
                        "responseType": "fillerId",
                        "confidence": 50.0,
                        "selected_id": None,
                        "selected_rank": None,
                        "threshold": getattr(decider, "threshold", np.nan),
                        "targetLineup": getattr(decider, "targetLineup", "unknown"),
                        "suspect": getattr(decider, "suspect", None),
                    })
                else:
                    # Choose the best candidate based on confidence
                    best = None
                    best_val = np.inf
                    for item in per_frame_candidates:
                        c = item["decision"].confidence
                        if c is None or (isinstance(c, float) and np.isnan(c)):
                            continue
                        if c < best_val:
                            best_val = c
                            best = item

                    # If no best candidate found, try to find the one with the minimum distance
                    if best is None:
                        # Directly use dist_map to find the best candidate
                        tmp_best_val = np.inf
                        tmp_best_item = None
                        for item in per_frame_candidates:
                            dm = item["dist_map"]
                            # Minimum distance value
                            vals = [v for v in dm.values() if
                                    v is not None and not (isinstance(v, float) and np.isnan(v))]
                            if len(vals) == 0:
                                continue
                            vmin = min(vals)
                            if vmin < tmp_best_val:
                                tmp_best_val = vmin
                                tmp_best_item = item
                        best = tmp_best_item

                    # Finally, if still no best candidate, we need to handle it
                    if best is None:
                        # No valid candidates found, reject this frame
                        frame_summary_rows.append({
                            "frame_index": frame_idx,
                            "time_sec": (frame_idx / fps) if fps > 0 else np.nan,
                            "has_faces": True,
                            "chosen_face_index": None,
                            "bbox_x1": None, "bbox_y1": None, "bbox_x2": None, "bbox_y2": None,
                            "bbox_w": None, "bbox_h": None,
                            "responseType": "rejectId",
                            "confidence": None,
                            "selected_id": None,
                            "selected_rank": None,
                            "threshold": getattr(decider, "threshold", np.nan),
                            "targetLineup": "unknown",
                            "suspect": getattr(decider, "suspect", None),
                        })
                    else:
                        d = best["decision"]
                        x1, y1, x2, y2, w, h = best["box"]
                        frame_summary_rows.append({
                            "frame_index": frame_idx,
                            "time_sec": (frame_idx / fps) if fps > 0 else np.nan,
                            "has_faces": True,
                            "chosen_face_index": best["i"],
                            "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": y2, "bbox_y2": y2,
                            "bbox_w": w, "bbox_h": h,
                            "responseType": d.responseType,
                            "confidence": d.confidence,
                            "selected_id": d.selected_id,
                            "selected_rank": d.selected_rank,
                            "threshold": d.threshold if hasattr(d, "threshold") else np.nan,
                            "targetLineup": d.targetLineup,
                            "suspect": d.suspect,
                        })

                frame_idx += 1
                processed += 1
                if pbar: pbar.update(1)
                if max_frames is not None and processed >= max_frames:
                    break
        finally:
            if pbar: pbar.close()
            cap.release()

        # save the results
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df_face = pd.DataFrame(face_rows)
        df_face.to_csv(output_csv, index=False)

        # save frame summary
        if output_frame_summary_csv is not None:
            os.makedirs(os.path.dirname(output_frame_summary_csv) or ".", exist_ok=True)
            df_frame = pd.DataFrame(frame_summary_rows)
            df_frame.to_csv(output_frame_summary_csv, index=False)

        return output_csv


class VerifyStyleDecider:
    """
    Pure virify decision maker: Automatically decides targetPresent/targetAbsent based on distances
    between fillers and perp/innocent, without using thresholds or targetLineup.
    Only requires knowing which row is the perp or innocent (if any).

    Parameters
    ----
    perp_index : Optional[str]
        The name of the row index representing the Perpetrator (if any).
    innocent_index : Optional[str]
        The name of the row index representing the Designated Innocent Suspect (if any).
    column_name : Optional[str]
        The column name in the similarity DataFrame to use for decision making.
    prefer_when_both : Literal["perp", "innocent"]
        A rare case where both perp and innocent are present. Decides which one to prefer.
    """
    def __init__(
        self,
        perp_index: Optional[str] = None,
        innocent_index: Optional[str] = None,
        column_name: Optional[str] = None,
        prefer_when_both: Literal["perp", "innocent"] = "perp"
    ):
        self.perp_index = perp_index
        self.innocent_index = innocent_index
        self.column_name = column_name
        self.prefer_when_both = prefer_when_both

    @staticmethod
    def _stable_rank(series: pd.Series, sel_id: str) -> Optional[int]:
        s = series.dropna().sort_values(kind="mergesort")  # 稳定排序
        if sel_id not in s.index:
            return None
        return int(s.rank(method="min").loc[sel_id])

    def decide(self, df: pd.DataFrame) -> LineupDecisionResult:
        if df is None or df.empty:
            raise ValueError("Empty similarity DataFrame.")

        col = self.column_name if self.column_name else df.columns[0]
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        # 当前列分数（越小越相似）
        scores = df[col].astype(float)

        # 标记身份是否在场
        perp_present = (self.perp_index is not None) and (self.perp_index in scores.index)
        innocent_present = (self.innocent_index is not None) and (self.innocent_index in scores.index)

        # 选出 fillers：排除 perp / innocent
        filler_scores = scores.drop(
            [x for x in [self.perp_index, self.innocent_index] if x is not None and x in scores.index],
            errors="ignore"
        )

        # 计算 fillerMin
        filler_scores_valid = filler_scores.replace([np.inf, -np.inf], np.nan).dropna()
        filler_min_id = None
        filler_min = np.nan
        if not filler_scores_valid.empty:
            filler_sorted = filler_scores_valid.sort_values(kind="mergesort")
            filler_min_id = filler_sorted.index[0]
            filler_min = float(filler_sorted.iloc[0])

        # 取 perp / innocent 的分数
        perp_dist = float(scores[self.perp_index]) if perp_present and pd.notna(scores[self.perp_index]) else np.nan
        innocent_dist = float(scores[self.innocent_index]) if innocent_present and pd.notna(scores[self.innocent_index]) else np.nan

        # 判定 TP/TA（自动）
        scenario: Literal["targetPresent", "targetAbsent", "unknown"]
        if perp_present and (not innocent_present or self.prefer_when_both == "perp"):
            scenario = "targetPresent"
            ref_id = self.perp_index
            ref_dist = perp_dist
            ref_as = "perp"
        elif innocent_present:
            scenario = "targetAbsent"
            ref_id = self.innocent_index
            ref_dist = innocent_dist
            ref_as = "innocent"
        else:
            scenario = "unknown"
            ref_id, ref_dist, ref_as = None, np.nan, None

        # 做出选择
        if scenario in ("targetPresent", "targetAbsent"):
            # 若填充值全 NaN 而参考分数有效 -> 选参考；若参考 NaN 而 fillerMin 有效 -> 选 filler
            if pd.notna(ref_dist) and (pd.isna(filler_min) or ref_dist < filler_min):
                # 选参考（perp 或 innocent）
                rtype = "suspectId" if scenario == "targetPresent" else "designateId"
                selected_id = ref_id
                confidence = ref_dist
            else:
                # 选 fillerMin（包括并列时稳定排序的第一个；或 ref_dist >= filler_min；或参考 NaN）
                rtype = "fillerId"
                selected_id = filler_min_id
                confidence = filler_min if pd.notna(filler_min) else np.nan
        else:
            # unknown：只能在 fillers 里选最小；若也没有有效分，则返回“拒认”
            if pd.notna(filler_min):
                rtype = "fillerId"
                selected_id = filler_min_id
                confidence = filler_min
            else:
                return LineupDecisionResult(
                    responseType="rejectId",
                    confidence=None,
                    selected_id=None,
                    selected_rank=None,
                    threshold=np.nan,
                    targetLineup="unknown",
                    suspect=None,
                    column_used=col,
                    extras={"reason": "no valid distances for any member"}
                )

        rank = self._stable_rank(scores.replace([np.inf, -np.inf], np.nan), selected_id) if selected_id else None

        return LineupDecisionResult(
            responseType=rtype,
            confidence=confidence,
            selected_id=selected_id,
            selected_rank=rank,
            threshold=np.nan,
            targetLineup=scenario,
            suspect=ref_id,
            column_used=col,
            extras={
                "perp_present": perp_present,
                "innocent_present": innocent_present,
                "perp_dist": perp_dist,
                "innocent_dist": innocent_dist,
                "filler_min_id": filler_min_id,
                "filler_min": filler_min,
                "ref_role": ref_as
            }
        )

