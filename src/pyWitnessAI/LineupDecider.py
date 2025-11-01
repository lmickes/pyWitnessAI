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
