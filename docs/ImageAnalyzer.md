
* **Overview**: Embedding backends (DeepFace vs FaceNet), supported distance metrics, device handling.
* **Public methods**: `process`, `process_verify`, `process_georgia_pipeline`, `dataframe`, `save`.
* **I/O shapes**: rows=lineup members, cols=probes (transposed to probe×member in pipeline).
* **Examples**: compare `euclidean` vs `cosine`; show effect of `align`/`normalization`.
* **Citations**: internal references to method signatures. 
