# VideoLineupPipeline

## 1. What is it?

`VideoLineupPipeline` is a high-level pipeline to process video files (or sequences of images) against a lineup of mugshots.
It combines face extraction, embedding computation, distance calculation, and lineup identification into a single workflow.

## 2. Usage

To use `VideoLineupPipeline`, you need to set up a `PipelineConfig` with your desired settings, 
including the face detection method, embedding model, distance metric, and lineup identifier settings. 
You can also set it within the `VideoLineupPipeline` constructor.

```python
from pyWitnessAI import *

config = PipelineConfig(
    analyzer_method="process_georgia_pipeline",
    detect_faces="mtcnn",
    no_face_fill=1.0,
    distance_metric="cosine",
    backend="facenet",
    model="Facenet512",
    align=True,
    normalization="Facenet",
    enforce_detection=False,
    show_progress=True,
    identifier_threshold=0.6,
    identifier_targetLineup=None,  # infer from roles
    identifier_target=None,         # infer from roles
)


```


* **Overview**: Frame loop, face extraction modes, handling no‑face frames, per‑frame probe naming.
* **Output schemas**: long vs wide; role‑min columns; export for pyWitness.
* **Config reference**: every `PipelineConfig` field with defaults.
* **Performance tips**: enable CUDA; choose `process_georgia_pipeline` for FaceNet embeddings; 
* batch size considerations (current API is per‑frame).


**Config (`PipelineConfig`):**

* `analyzer_method`: `"process" | "process_verify" | "process_georgia_pipeline"`
* `distance_metric`, `backend`, `model`, `align`, `normalization`, `enforce_detection`, `show_progress`
* `detect_faces`: `"fullframe"` or `"mtcnn"`
* `no_face_fill`: distance to write when no face is found (useful with `mtcnn`). 

**Face extraction per frame:**

* `fullframe`: treat the whole frame as a single face (fast baseline).
* `mtcnn`: detect & crop; if none found, returns `[]`. 

**No‑face handling:** if no face is detected in a frame, the pipeline still emits rows for each lineup member 
with `distance=no_face_fill` and (if identifier enabled) a filler‑like summary.

**Output forms:**

* **Long form** (internal): each record is `(frame, probe, lineup_member, distance, role, …summary)`
* **Wide form**: `pivot_table` view with optional role‑min columns via `_long_to_wide(...)`. 
* `export_for_pywitness(...)`: utility to turn a long‑form CSV with `TA_*`/`TP_*` into a compact file for pyWitness (`responseType, confidence, targetLineup, lineupSize`). fileciteturn0file3
