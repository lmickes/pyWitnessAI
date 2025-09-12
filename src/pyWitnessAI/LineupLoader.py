import os
import glob
import random
import logging
from PIL import Image
from typing import List, Optional, Dict, Literal, Tuple

from pyWitnessAI import VerifyStyleDecider


class LineupLoader:
    def __init__(self, folder_path: str, guilty_suspect: Optional[str] = None, innocent_suspect: Optional[str] = None):
        self.folder_path = folder_path
        self.guilty_suspect = guilty_suspect
        self.innocent_suspect = innocent_suspect

        '''
        Example of image_groups structure:
        { "guilty_suspect": ["/path/Video1_Perpetrator.png"],
          "innocent_suspect": ["/path/Video1_Innocent.png"],
          "filler": ["/path/Filler1.png", "/path/Filler2.png", ...]}
        '''
        self.image_groups = self._load_and_classify(folder_path)

        # Name mapping for easy reference
        self.names: Dict[str, str] = {}
        for p in sum(self.image_groups.values(), []):
            base_name = os.path.splitext(os.path.basename(p))[0]
            self.names[base_name] = p

        self.perp_name: Optional[str] = self._first_name("guilty_suspect")
        self.innocent_name: Optional[str] = self._first_name("innocent_suspect")
        self.filler_names: List[str] = [os.path.splitext(os.path.basename(p))[0]
                                        for p in self.image_groups.get("filler", [])]

        # Error when neither exists
        if not self.image_groups["guilty_suspect"] and not self.image_groups["innocent_suspect"]:
            raise ValueError("Missing guilty suspect image. Please rename the perpetrator image or provide it.")

        # Default lineup (if perp -> TP; otherwise ->TA)
        self.default_target: Literal["targetPresent", "targetAbsent"] = \
            ("targetPresent" if self.perp_name else "targetAbsent")

        self.lineup: List[str] = self.generate_lineup()

    def _load_and_classify(self, folder_path: str) -> Dict[str, List[str]]:
        extensions = ["jpg", "jpeg", "png", "gif", "webp", "bmp"]
        # image_paths = glob.glob(os.path.join(folder_path, "*.png"))
        image_paths = sum(
            [glob.glob(os.path.join(folder_path, f"*.{ext}")) +
             glob.glob(os.path.join(folder_path, f"*.{ext.upper()}"))
             for ext in extensions], []
        )
        image_dict = {"guilty_suspect": [], "innocent_suspect": []}

        for image_path in image_paths:
            name = os.path.basename(image_path).lower()
            if "perp" in name:
                image_dict["guilty_suspect"].append(image_path)
                self.guilty_suspect = image_path
            elif "designated" in name or "inn" in name:
                image_dict["innocent_suspect"].append(image_path)
                self.innocent_suspect = image_path
            else:
                image_dict["filler"].append(image_path)

        # !!!!!!!! Solve the problem of guilty suspect, maybe join with folder path?
        if self.guilty_suspect:
            image_dict["guilty_suspect"] = [self.guilty_suspect]
        if self.innocent_suspect:
            image_dict["innocent_suspect"] = [self.innocent_suspect]

        logging.debug(f"Guilty suspect image: {image_dict['guilty_suspect']}")
        logging.debug(f"Innocent suspect image: {image_dict['innocent_suspect']}")
        logging.debug(f"Number of fillers: {len(image_dict['filler'])}")
        return image_dict

    def _first_name(self, category: str) -> Optional[str]:
        paths = self.image_groups.get(category, [])
        if not paths:
            return None
        return os.path.splitext(os.path.basename(paths[0]))[0]

    def generate_lineup(self, lineup_size: int = 6,
                        target_lineup: Optional[Literal["targetPresent", "targetAbsent"]] = None,
                        shuffle: bool = False) -> List[str]:
        target = target_lineup or self.default_target

        if target == "targetAbsent":
            need_fillers = lineup_size - (1 if self.innocent_name else 0)
        else:
            need_fillers = lineup_size - 1

        if len(self.image_groups["filler"]) < need_fillers:
            raise ValueError("Insufficient fillers to generate lineup.")

        lineup: List[str] = []
        if target_lineup == "targetPresent":
            assert self.image_groups["guilty_suspect"], "Guilty suspect image is required for targetPresent lineup."
            lineup.append(self.image_groups["guilty_suspect"][0])
        else:
            assert self.image_groups["innocent_suspect"], "Innocent suspect image is required for targetAbsent lineup."
            lineup.append(self.image_groups["innocent_suspect"][0])

        fillers = list(self.image_groups["filler"])  #  Copy to avoid modifying original

        lineup += fillers[:need_fillers]
        if shuffle:
            random.shuffle(lineup)

        self.lineup = lineup[:lineup_size]  # Ensure correct size
        return lineup[:lineup_size]

    def show_lineup(self, rows: int = 2, cols: int = 3):
        if len(self.lineup) > rows * cols:
            raise ValueError("Number of images exceeds grid capacity.")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(cols * 3, rows * 3))
        for idx, image_path in enumerate(self.lineup):
            img = Image.open(image_path)
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()
