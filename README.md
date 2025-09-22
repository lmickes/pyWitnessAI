# pyWitnessAI

> A practical pipeline to turn videos + lineup photos into frame‑wise similarity matrices and lineup decisions compatible with pyWitness.

- **Modules covered:** `ImageAnalyzer`, `LineupLoader`, `LineupIdentifier`, `VideoLineupPipeline`
- **Who is this for?** Researchers building AI‑assisted police lineup experiments, or anyone needing consistent, reproducible frame‑level identification summaries.

---

## 1) Quick Start

### Install

```bash
pip install "pyWitnessAI"
```

### Minimal example

```python
from pyWitnessAI import *

# Build a lineup from a folder of images
loader = LineupLoader("/path/to/lineup_folder")

# Run on a video file, get similarity score (distance) with decisions
pipe = VideoLineupPipeline(
    video_path="/path/to/crime_video.mp4",
    lineup_loader=loader,
    cfg={"analyzer_method":"process", model="Facenet", detect_faces="mtcnn", "show_progress":True},
    identifier=True,  # True | False | LineupIdentifier
)

df = pipe.run(output_csv="results/run1.csv", export_pywitness=True)
print(df.head())
```

The pipeline outputs a DataFrame (and a CSV if requested) containing per‑frame/per‑probe distances 
to lineup members plus optional summary columns from the identifier.
When `export_pywitness=True`, a pyWitness‑readable `*_pywitness.csv` is also written.

---

## 2) Concepts at a Glance

- When `LineupLoader` is called, **roles** are assigned to lineup members, 
such as guilty suspect, innocent suspect, or filler.
- `ImageAnalyzer` will calculate the **distance** (aka. similarity score, lower is more similar). 
Distances are computed between a probe face from each frame and each lineup member face.
- `LineupIdentifier` turns per‑frame similarity matrices into lineup decisions that can be interpreted by pyWitness.
- `VideoLineupPipeline` orchestrates video faces & lineup faces → similarities → decisions (optional) → CSVs. 

---

## 3) Module Reference

### 3.1 `ImageAnalyzer`

**Purpose:** compute pairwise distances between two image sets: **rows** (can be lineup members) 
and **columns** (can be probes per frame).

**Key features:**

- Supports embedding extraction using **[DeepFace](https://github.com/serengil/deepface)** (`process`, `process_verify`) 
or alternatively the MTCNN combined with Facenet approach, consistent with the methodology 
described by [Kleider-Offutt et al (2024)](https://doi.org/10.1186/s41235-024-00542-0) 
  (`process_georgia_pipeline`).
- Returns a `pandas.DataFrame` similarity matrix and can persist to CSV via `save()`.

**Constructor (abridged):**

```python
ImageAnalyzer(
    column_images: ImageLoader,  # probes (per-frame faces)
    row_images: ImageLoader,     # lineup members
    show_progress: bool = False,
    distance_metric: str = "euclidean", # Deepface param
    backend: str = "mtcnn",             # Deepface param
    enforce_detection: bool = False,    # Deepface param
    model: str = "Facenet",             # Deepface param
    align: bool = False,                # Deepface param
    normalization: str = "base",        # Deepface param
)
```

Parameters with `# Deepface param` are those passed to [DeepFace](https://github.com/serengil/deepface)’s `represent` or `verify` functions.

**Typical usage**

```python
analyzer = ImageAnalyzer(column_images=probes, row_images=lineup_faces)
analyzer.process()  # process / process_verify / process_georgia_pipeline
sim_df = analyzer.dataframe()
```

---

### 3.2 `LineupLoader`

**Purpose:** read a folder of images and classify them into roles.

**Key features:**
- Classifies images into `guilty_suspect`, `innocent_suspect`, or `filler` based on filename heuristics or explicit paths.
- Can generate a lineup of specified size and condition.
- Can visualize the lineup in a grid.

**Classification rules:**

- If explicit paths are provided for `guilty_suspect` or `innocent_suspect`, those win.
- Otherwise:
  - **filenames** containing "perp" → `guilty_suspect`; 
  - **filenames** containing "designated" or "innocent" → `innocent_suspect`;
  - all others → `filler`.
- If neither guilty nor innocent exists, users will be asked to provide at least one suspect.

**Quick snippet:**

```python
loader = LineupLoader("/path/to/lineup_mambers", guilty_suspect=None, innocent_suspect=None)
print(loader.perp_name, loader.innocent_name, loader.filler_names)
print(loader.lineup)  # list of image paths in lineup order
```

---

### 3.3 `LineupIdentifier`

**Purpose:** convert a similarity matrix of **target face x lineup members** into decisions. 
In video pipeline mode, it helps turn frame-level similarity matrices into lineup decisions.

**Key features:**
- Outputs a compact summary dict with selection, response type, and confidence.
  - **Response types:** `suspectId`, `designateId`, `fillerId`, `rejectId`. These are in line with pyWitness expectations.

**How [roles](README.md#2-concepts-at-a-glance) are sourced:**

- In general case, if you only have a similarity matrix, roles can be built in constructor via `target` (suspect name) and `targetLineup` (condition).
- In video pipeline mode, roles come from the associated `LineupLoader`.

**Decision strategies:** 

It supports different strategies depending on the presence of suspects and settings, the identifier can produce:
  - **macro** decisions when both target‑present and target‑absent in one pass,
  - or **targetPresent** / **targetAbsent** decisions for a specific suspect.

See **[docs/LineupIdentifier.md](docs/LineupIdentifier.md)** for API, examples, and integration notes.

**Quick snippet:**

```python
sim_df = pd.DataFrame({"filler1.jpg": [0.8], "filler2.jpg": [0.6], "filler3.jpg": [0.4]}, index=["frame0_targetface"])
identifier = LineupIdentifier(threshold=0.5, targetLineup="targetPresent", target="target")
summary = identifier.decide(sim_df)  
```
---

### 3.4 `VideoLineupPipeline`

**Purpose:** tie it all together: extract faces from each video frame; compute distances to lineup members; 
optionally compute decisions; emit tidy CSVs.

**Key features:**
- Outputs a DataFrame/CSV, with role-min summary columns.
- Can export a compact CSV compatible with pyWitness via `export_for_pywitness`.

**Run API:**

```python
pipe = VideoLineupPipeline(
    video_path="/path/to/crime_video.mp4",
    lineup_loader=LineupLoader("/path/to/lineup_folder"),
    cfg={"distance_metric": "euclidean", "backend": "mtcnn", "model": "Facenet", "show_progress": False,}
    identifier=True
)
df = pipe.run(output_csv="results/run1.csv", export_pywitness=True)
```

- When `export_pywitness=True`, the function writes an additional `*_pywitness.csv`.

---

## 4) End‑to‑End Examples

### A. TP/TA **macro** decisions in one pass

```python
loader = LineupLoader("/data/lineup", guilty_suspect="Perpetrator", innocent_suspect="Innocent")
idf = LineupIdentifier(threshold=1.0)
cfg = {"model"="Facenet", "detect_faces"="mtcnn"}

pipe = VideoLineupPipeline("/data/video.mp4", loader, cfg, identifier=idf)
df = pipe.run(output_csv="result.csv", export_pywitness=True)
```

- The output CSV *result.csv* contains columns such as `TA_selected`, `TP_selected`, `TA_responseType`, `TP_responseType`, 
`TA_conf`, `TP_conf`, plus per‑member distances and `min_by_role[...]` columns. 
- The output CSV *result_pywitness.csv* contains `responseType, confidence, targetLineup, lineupSize` for pyWitness.

### B. Force **targetPresent** decisions for a specific suspect

```python
idf_tp = LineupIdentifier(threshold=1.0, targetLineup="targetPresent", target="Perpetrator")
pipe_tp = VideoLineupPipeline("/data/video.mp4", loader, cfg, idf_tp)
df_tp = pipe_tp.run(output_csv="result_tp.csv", export_pywitness=True)
```

Debugging in progress: There are some logical loopholes while we set "fullframe" and target condition at the same time.

### C. Distances only (no identifier)

```python
pipe = VideoLineupPipeline("/data/video.mp4", loader, PipelineConfig(), identifier=False)
df = pipe.run(output_csv="distances_only.csv", wide=True)
```

---

## 5) File Outputs

* `result.csv`: per‑frame/per‑probe distances; optional summary columns (TA/TP).
* `result_pywitness.csv` — compact table for pyWitness: `responseType, confidence, targetLineup, lineupSize`.
Generated when set `export_pywitness` as **True**. 


## FAQ

**Q: My similarity matrix is empty?**

A: Call an analyzer method first (`process*`) before `dataframe()`. 

**Q: How are decisions made if I don’t set a `targetLineup`?**

A: If both suspects exist, identifier populates TA & TP fields at once; 
otherwise a simple min‑choice path is used. 

**Q: What happens when MTCNN finds no face in a frame?**

A: The pipeline writes rows for every lineup member using `no_face_fill` (50 as default)
as the distance and adds a filler‑style summary if identifier is enabled. 

---

