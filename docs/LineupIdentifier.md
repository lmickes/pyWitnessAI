# LineupIdentifier

## 1. What is it?

`LineupIdentifier` 

## 2. Usage

To initialize a `LineupIdentifier`, you should provide: Threshold, targetLineup (optional), and target (optional).
These are the parameters that govern how the decision is made.

- `threshold`: the minimum distance required to select a lineup member. If the minimum distance is above this threshold, 
the identifier will return "rejectId".
- `targetLineup`: `"targetPresent" | "targetAbsent" | None`. This indicates whether the lineup is target-present 
or target-absent. If `None`, the identifier will try to infer the context from the roles.
- `target`: the explicit name of the suspect (guilty or innocent). 
This is optional and can be used to override the default behavior.

Now we can apply those settings to make decisions based on similarity matrices.

**Example similarity matrix:**

|               | member_1 | member_2 | member_3 | member_4 | member_5 | member_6 |
|---------------|----------|----------|----------|----------|----------|----------|
| probe_face_1  | 0.2      | 0.5      | 0.8      | 0.3      | 0.8      | 0.3      |
| probe_face_2  | 0.6      | 0.4      | 0.9      | 0.1      | 0.9      | 0.1      |

  - typically lower distance = more similar 
  - Rows: probe face(s) from the current frame 
  - Columns: lineup members (filenames/base names)

- a role map derived from `LineupLoader`:
    `{"member_name": "guilty_suspect" | "innocent_suspect" | "filler"}`

- Output: a compact decision summary (min distance, response type, confidence), optionally for 
**target-present (TP)** and **target-absent (TA)** simultaneously.

## 3. Strategies

- `decide_macro(sim_df, roles)`: Use when both guilty and innocent exist and lineup condition (TA or TP) is **not** set. 
 Returns the minimum distance among fillers, TA and TP selections, TA and TP response types, and TA and TP confidences. 
- `decide_tp_or_ta(sim_df, roles)`: Use when lineup condition is specified. 
Returns selection and confidence for that condition.
- `decide_for_pipeline(sim_df, lineup_loader)`: Auto‑select strategy for the pipeline based on what exists in the loader and specify in settings.
- `decide(sim_df)`: General interface when you only pass a matrix, **not** with a lineuploader-loaded lineup.
In this general case, both `targetLineup` and `target` should be initialized in the constructor.


Return schemas: the dict shape for each strategy; how `filler_min` is computed.
Integration: used inside pipeline’s `_decide_summary`.
