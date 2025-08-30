from math import inf
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd


# -------------------------
# 基本ユーティリティ
# -------------------------
def z_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    if sigma < eps:
        return x - mu  # zero-variance の場合は平均引きのみ
    return (x - mu) / sigma


def sliding_windows(arr: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """arr: 1D numpy, 戻り: shape=(n_windows, window)"""
    n = arr.shape[0]
    if window > n:
        return np.empty((0, window))
    num = 1 + (n - window) // stride
    # faster approach using strides (works for contiguous) or fallback
    out = np.lib.stride_tricks.sliding_window_view(arr, window)[::stride]
    return out


# -------------------------
# 距離計算
# -------------------------
def euclidean_distances(query: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """windows: (m, L), query: (L,) -> L2 norm per window"""
    dif = windows - query.reshape(1, -1)
    return np.linalg.norm(dif, axis=1)


def cosine_distances(query: np.ndarray, windows: np.ndarray) -> np.ndarray:
    qnorm = np.linalg.norm(query)
    wnorm = np.linalg.norm(windows, axis=1)
    denom = qnorm * wnorm
    # avoid div0
    denom = np.where(denom == 0, 1e-8, denom)
    cos_sim = (windows @ query) / denom
    # convert to distance
    return 1.0 - cos_sim


def dtw_distance(a: np.ndarray, b: np.ndarray, window: Optional[int] = None) -> float:
    """Simple DTW (O(N*M)). window: Sakoe-Chiba radius (int) to constrain band."""
    n, m = len(a), len(b)
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))
    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jmin = max(1, i - window)
        jmax = min(m, i + window)
        for j in range(jmin, jmax + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]


# -------------------------
# メイン：類似区間検索
# -------------------------
def find_similar_subsequences(
    series: pd.Series,
    query_slice: Union[slice, Tuple[Union[int, str], Union[int, str]]],
    window: Optional[int] = None,
    stride: int = 1,
    method: Literal["euclid", "cosine", "dtw"] = "euclid",
    znorm: bool = True,
    top_k: int = 5,
    threshold: Optional[float] = None,
    dtw_window: Optional[int] = None,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """
    series: pandas.Series (index preserved)
    query_slice: slice or (start, end)
    window: length of sliding window; if None, uses len(query)
    method: distance method
    top_k: number of top matches to return (if threshold is None)
    threshold: if given, return all with distance <= threshold
    exclude_self: クエリ区間自身と重複する候補を除外
    """
    s = series.dropna()  # 欠損は一旦削る（別扱い必要なら拡張）
    arr = s.values
    idx = s.index

    # resolve query slice to integer positions
    if isinstance(query_slice, slice):
        # pandas slice may accept labels; use index.get_loc if necessary
        try:
            start = s.index.get_loc(query_slice.start) if query_slice.start is not None else 0
        except Exception:
            start = query_slice.start or 0
        try:
            stop = (
                s.index.get_loc(query_slice.stop) + 1 if query_slice.stop is not None else len(arr)
            )
        except Exception:
            stop = query_slice.stop or len(arr)
    elif isinstance(query_slice, tuple) and len(query_slice) == 2:
        a, b = query_slice
        try:
            start = s.index.get_loc(a) if a is not None else 0
        except Exception:
            start = int(a)
        try:
            stop = s.index.get_loc(b) + 1 if b is not None else len(arr)
        except Exception:
            stop = int(b) + 1
    else:
        raise ValueError("query_slice should be slice or (start, end)")

    query = arr[start:stop]
    L = len(query)
    if L == 0:
        raise ValueError("query interval is empty")
    if window is None:
        window = L
    if window > len(arr):
        raise ValueError("window > length of series")

    # build windows
    win_arr = sliding_windows(arr, window=window, stride=stride)  # shape (m, window)
    # compute start indices of each window
    n_windows = win_arr.shape[0]
    starts = np.arange(0, n_windows * stride, stride)
    # exclude windows that overlap the query if requested
    if exclude_self:
        overlap_mask = ~(((starts + window) <= start) | (starts >= stop))
        valid_mask = ~overlap_mask
    else:
        valid_mask = np.ones(n_windows, dtype=bool)

    # prepare query for comparison: if query length != window, either center/crop or pad
    if L != window:
        # simplest: if query shorter -> center-pad with its mean; if longer -> center-crop
        if L < window:
            pad = window - L
            left = pad // 2
            right = pad - left
            padvals = np.full((left,), np.nan)  # better to pad with nan then fill with mean
            q_full = np.concatenate([padvals, query, np.full((right,), np.nan)])
            # replace nan by mean of query
            q_full = np.where(np.isnan(q_full), np.nanmean(query), q_full)
        else:
            # crop center
            startq = (L - window) // 2
            q_full = query[startq : startq + window]
    else:
        q_full = query.copy()

    if znorm:
        q_full = z_normalize(q_full)
        # znorm windows rowwise
        # avoid repeated loops: compute mean/std per row
        win_mean = np.mean(win_arr, axis=1, keepdims=True)
        win_std = np.std(win_arr, axis=1, keepdims=True)
        win_std[win_std == 0] = 1.0
        win_z = (win_arr - win_mean) / win_std
        comp_windows = win_z
    else:
        comp_windows = win_arr

    # compute distances
    if method == "euclid":
        distances = euclidean_distances(q_full, comp_windows)
    elif method == "cosine":
        distances = cosine_distances(q_full, comp_windows)
    elif method == "dtw":
        # slow: compute dtw per candidate (respect valid_mask)
        distances = np.full(n_windows, np.inf)
        for i in range(n_windows):
            if not valid_mask[i]:
                continue
            distances[i] = dtw_distance(q_full, comp_windows[i], window=dtw_window)
    else:
        raise ValueError("unknown method")

    # mask invalid
    distances = np.where(valid_mask, distances, np.inf)
    # build DataFrame of results
    res = pd.DataFrame({"start_pos": starts, "end_pos": starts + window - 1, "distance": distances})
    # map to original index labels if possible
    res["start_index"] = [idx[p] for p in res["start_pos"]]
    res["end_index"] = [idx[p] for p in res["end_pos"]]
    # filter by threshold or top_k
    if threshold is not None:
        sel = res[res["distance"] <= threshold].sort_values("distance")
    else:
        sel = res.sort_values("distance").head(top_k)
    # attach subsequences (as numpy arrays) if desired
    sel = sel.reset_index(drop=True)
    sel["subsequence"] = [
        arr[int(r["start_pos"]) : int(r["end_pos"]) + 1].copy() for _, r in sel.iterrows()
    ]
    return sel


# -------------------------
# 異常検出ルール群（ご要望のルールを実装）
# -------------------------
def detect_anomalies_by_rules(series: pd.Series) -> dict:
    """
    returns dict of lists of indices for each rule found.
    Rules implemented:
      - 'same_side_mean_10': 平均値に対して同じ側に10点連続
      - 'monotonic_5': 5連続増加 or 減少
      - 'cross_mean_10': 10連続で平均値を横切る（各点が平均を超える/未満を連続で切り替える）
    """
    arr = series.values
    idx = series.index
    n = len(arr)
    res = {"same_side_mean_10": [], "monotonic_5": [], "cross_mean_10": []}
    mean = np.nanmean(arr)
    # same_side_mean_10
    side = np.sign(arr - mean)  # -1,0,1
    # treat 0 as same side as previous non-zero (or break)
    run_val = None
    run_len = 0
    run_start = 0
    for i, s in enumerate(side):
        if s == 0:
            # break the run
            run_val = None
            run_len = 0
            continue
        if run_val is None or s != run_val:
            run_val = s
            run_len = 1
            run_start = i
        else:
            run_len += 1
        if run_len >= 10:
            res["same_side_mean_10"].append(idx[run_start : i + 1])
            # move on (don't double-count overlapping runs) -> skip ahead
            run_val = None
            run_len = 0

    # monotonic_5 (increase or decrease)
    run_dir = None
    run_len = 1
    run_start = 0
    for i in range(1, n):
        if np.isnan(arr[i]) or np.isnan(arr[i - 1]):
            run_dir = None
            run_len = 1
            continue
        if arr[i] > arr[i - 1]:
            d = 1
        elif arr[i] < arr[i - 1]:
            d = -1
        else:
            d = 0
        if d == run_dir and d != 0:
            run_len += 1
        else:
            run_dir = d if d != 0 else None
            run_len = 2 if d != 0 else 1
            run_start = i - 1
        if run_len >= 5:
            res["monotonic_5"].append(idx[run_start : i + 1])
            run_dir = None
            run_len = 1

    # cross_mean_10: 連続10点で平均値を交差し続ける（signが交互に変わる）
    signs = np.sign(arr - mean)
    # remove zeros by treating as previous non-zero where possible
    for i in range(1, n):
        if signs[i] == 0:
            signs[i] = signs[i - 1] if signs[i - 1] != 0 else 0
    # check alternating pattern length >= 10 where sign flips each time
    run_len = 1
    run_start = 0
    for i in range(1, n):
        if signs[i] == 0 or signs[i - 1] == 0:
            run_len = 1
            run_start = i
            continue
        if signs[i] == -signs[i - 1]:
            run_len += 1
        else:
            run_len = 1
            run_start = i
        if run_len >= 10:
            res["cross_mean_10"].append(idx[run_start : i + 1])
            run_len = 1
            run_start = i + 1
    return res


# -------------------------
# 異常発生時に過去の類似点を探すヘルパー
# -------------------------
def find_similar_to_anomaly(
    series: pd.Series,
    anomaly_window_center: Union[int, pd.Timestamp],
    pattern_len: int,
    search_end_pos: Optional[int] = None,
    **find_kwargs,
) -> pd.DataFrame:
    """
    anomaly_window_center: 中心位置（integer pos or timestamp label）
    pattern_len: パターン長（window）
    search_end_pos: 検索を現時点より過去に限定する場合の終端位置（exclusive）
    その他は find_similar_subsequences に渡る
    """
    s = series.dropna()
    idx = s.index
    # map center to integer pos
    try:
        center_pos = s.index.get_loc(anomaly_window_center)
    except Exception:
        center_pos = int(anomaly_window_center)
    half = pattern_len // 2
    start = max(0, center_pos - half)
    stop = min(len(s), start + pattern_len)
    query_slice = (idx[start], idx[stop - 1])
    # limit search range by cropping series up to search_end_pos if provided
    if search_end_pos is not None:
        try:
            endpos = s.index.get_loc(search_end_pos)
        except Exception:
            endpos = int(search_end_pos)
        ser_search = s.iloc[:endpos]
    else:
        ser_search = s
    return find_similar_subsequences(
        ser_search, query_slice=query_slice, window=pattern_len, **find_kwargs
    )
