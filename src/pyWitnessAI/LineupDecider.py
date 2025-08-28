from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import pandas as pd
import numpy as np

ResponseType = Literal["suspectId", "fillerId", "rejectId", "designateId"]

@dataclass
class LineupDecisionResult:
    responseType: ResponseType
    confidence: Optional[float]              # The distance score of the selected object; None if rejected
    selected_id: Optional[str]               # Index of the selected object in the DataFrame; None if rejected
    selected_rank: Optional[int]             # 1-based rank of the selected object; None if rejected
    threshold: float
    targetLineup: Literal["targetPresent", "targetAbsent"]
    suspect: Optional[str]                   # Index of the suspect image; None if not applicable
    column_used: str                         # The column name of the dataframe used for decision
    extras: Dict[str, Any]                   # Extra information, e.g., sorted scores, reason for rejection, etc.


class LineupDecider:
    """
    Use a similarity DataFrame to decide the response type for a lineup.
    The DataFrame should have the following structure:
    - Index: Object IDs (e.g., image names)
    - Columns: Similarity scores (e.g., distances) for each object
    """
    def __init__(
            self,
            targetLineup: Literal["targetPresent", "targetAbsent"],
            suspect: Optional[str],
            threshold: float = 1.0,
            column_name: Optional[str] = None
    ):
        self.targetLineup = targetLineup
        self.suspect = suspect    # Index of the suspect image; None if not applicable
        self.threshold = threshold
        self.column_name = column_name    # If None, use the first column

    def decide(self, df: pd.DataFrame) -> LineupDecisionResult:
        if df is None or df.empty:
            raise ValueError("Empty similarity DataFrame.")

        col = self.column_name if self.column_name else df.columns[0]
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        # the column should contain similarity scores (e.g., distances)
        scores = df[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if scores.empty:
            # If all scores are NaN or invalid, reject the lineup
            return LineupDecisionResult(
                responseType="rejectId",
                confidence=None,
                selected_id=None,
                selected_rank=None,
                threshold=self.threshold,
                targetLineup=self.targetLineup,
                suspect=self.suspect,
                column_used=col,
                extras={"reason": "all NaN/invalid scores"}
            )

        # Sort scores to find the best match
        order = scores.sort_values(kind="mergesort")  # Ranking stable sort
        best_id = order.index[0]
        best_score = float(order.iloc[0])

        # Calculate the rank of the best score
        ranks = order.rank(method="min")  # Use minimum rank for ties
        best_rank = int(ranks.loc[best_id])

        # Check if the best score meets the threshold
        if not (best_score < self.threshold):
            return LineupDecisionResult(
                responseType="rejectId",
                confidence=None,
                selected_id=None,
                selected_rank=None,
                threshold=self.threshold,
                targetLineup=self.targetLineup,
                suspect=self.suspect,
                column_used=col,
                extras={"best_id": best_id, "best_score": best_score, "all_scores_sorted": order.to_dict()}
            )

        # Determine the response type based on the target lineup and best match
        if self.targetLineup == "targetPresent":
            # Normal case: suspect is present in the lineup
            rtype: ResponseType = "suspectId" if (self.suspect is not None and best_id == self.suspect) else "fillerId"
        elif self.targetLineup == "targetAbsent":
            # Special case: suspect is absent from the lineup
            rtype = "designateId" if (self.suspect is not None and best_id == self.suspect) else "fillerId"
        else:
            raise ValueError("targetLineup must be 'targetPresent' or 'targetAbsent'.")

        return LineupDecisionResult(
            responseType=rtype,
            confidence=best_score,
            selected_id=best_id,
            selected_rank=best_rank,
            threshold=self.threshold,
            targetLineup=self.targetLineup,
            suspect=self.suspect,
            column_used=col,
            extras={"all_scores_sorted": order.to_dict()}
        )

