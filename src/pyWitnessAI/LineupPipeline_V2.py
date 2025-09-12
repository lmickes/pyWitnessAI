import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Optional, Literal
from dataclasses import dataclass

from ImageAnalyzer import ImageLoader as IAImageLoader, ImageAnalyzer
from LineupLoader import LineupLoader
from


@dataclass
class PipelineConfig:
    analyzer_method: Literal["process", "process_verify", "process_georgia_pipeline"] = "process_georgia_pipeline"
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
      - detect/crop faces -> column_images
      - lineup images -> row_images
      - run ImageAnalyzer.<method>()
      - (optional) Identifier to summarize
      - append to a single CSV (long-form + per-frame summary columns)
    """

    def __init__(self,
                 video_path: str,
                 lineup_loader: LineupLoader,
                 cfg: PipelineConfig = PipelineConfig(),
                 identifier: Optional[Identifier] = None):
        self.video_path = video_path
        self.lineup_loader = lineup_loader
        self.cfg = cfg
        self.identifier = identifier

        # lineup row
        # 行：由 lineup_loader.lineup 的“文件路径列表”构造
        self._row_il = IAImageLoader([*self.lineup_loader.lineup])  # paths -> PIL
        # 角色字典（供写 CSV）
        self.roles = Identifier._role_map_from_lineuploader(self.lineup_loader)

    def _faces_from_frame(self, frame_bgr: np.ndarray) -> List[Image.Image]:
        """
        使用 facenet-pytorch 的 MTCNN（通过 ImageAnalyzer 内部）进行对齐/裁剪也可，
        这里用一个简化的灰度 + 人脸检测占位（你可替换为更稳健的 MTCNN 裁剪）。
        """
        # 简化示例：直接把整帧当作“一个 probe”（若你已有更精准的 crop，可替换）
        # 为了与 ImageAnalyzer 对接，这里返回 PIL.Image 列表
        img_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        return [pil]  # 你可以替换为“多脸列表”

    def run(self, output_csv: str) -> str:
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        records = []  # 每条记录：一对 (frame_idx, probe_name, lineup_member, distance, role, …摘要…)
        frame_idx = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            faces = self._faces_from_frame(frame)  # List[PIL.Image]
            if len(faces) == 0:
                # 无脸帧：为每个 lineup 成员写一行 distance=50，并填入固定摘要
                for lm_path in self.lineup_loader.lineup:
                    lm_name = lm_path.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
                    role = self.roles.get(lm_name, "filler")
                    records.append({
                        "frame": frame_idx,
                        "probe": None,
                        "lineup_member": lm_name,
                        "distance": 50.0,
                        "role": role,
                        # 摘要（无脸规则）
                        "filler_min": 50.0,
                        "selected": None,
                        "responseType": "fillerId",
                        "confidence": 50.0,
                        # 宏观字段占位（若需要）
                        "TA_selected": None, "TP_selected": None,
                        "TA_responseType": None, "TP_responseType": None,
                        "TA_conf": None, "TP_conf": None,
                    })
                continue

            # 构造 column_images（以该帧检测到的人脸序列）
            probe_names = [f"frame{frame_idx}_face{i}" for i in range(len(faces))]
            col_il = IAImageLoader([])     # 空初始化后填充
            col_il.images = {n: im for n, im in zip(probe_names, faces)}
            col_il.path_to_images = {n: f"{n}.png" for n in probe_names}  # 虚拟路径名以兼容接口

            # 运行 ImageAnalyzer
            analyzer = ImageAnalyzer(
                column_images=col_il,
                row_images=self._row_il,
                distance_metric=self.cfg.distance_metric,
                backend=self.cfg.backend,
                enforce_detection=self.cfg.enforce_detection,
                model=self.cfg.model,
                align=self.cfg.align,
                normalization=self.cfg.normalization,
                show_progress=self.cfg.show_progress
            )
            getattr(analyzer, self.cfg.analyzer_method)()  # e.g., 'process_georgia_pipeline'
            sim_df = analyzer.dataframe()  # 行=阵容，列=本帧人脸

            # Identifier（若提供）
            summary = {
                # 默认空；下面根据场景填入
                "filler_min": None, "selected": None, "responseType": None, "confidence": None,
                "TA_selected": None, "TP_selected": None,
                "TA_responseType": None, "TP_responseType": None,
                "TA_conf": None, "TP_conf": None,
            }
            if self.identifier is not None:
                summ = self.identifier.decide(sim_df, self.lineup_loader)
                summary.update(summ)

            # 写入“长表 + 摘要列”
            per_member_min = sim_df.min(axis=1)
            for lm_name, row in sim_df.iterrows():
                role = self.roles.get(lm_name, "filler")
                for probe, dist in row.items():
                    rec = {
                        "frame": frame_idx,
                        "probe": probe,
                        "lineup_member": lm_name,
                        "distance": float(dist),
                        "role": role,
                        # 摘要（同一 frame 固定）
                        **summary
                    }
                    records.append(rec)

        cap.release()
        df = pd.DataFrame.from_records(records)

        # 单一 CSV：包含所有逐帧比对 + 每帧统一摘要列
        df.to_csv(output_csv, index=False)
        return output_csv
