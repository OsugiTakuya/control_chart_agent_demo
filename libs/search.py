import datetime
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
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


def check_cols_and_ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"login_date", "product", "parameter", "value"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"df must contain columns: {required_cols}")

    if not pd.api.types.is_datetime64_any_dtype(df["login_date"]):
        df = df.copy()
        df["login_date"] = pd.to_datetime(df["login_date"])
    return df


def find_start_end_index(
    df: pd.DataFrame, target_time: datetime.datetime, window: int
) -> Tuple[int, int]:
    le_mask = df["login_date"] <= pd.to_datetime(target_time)
    if not le_mask.any():
        raise ValueError("No data at or before target_time for the source series.")
    end_idx = df[le_mask].index.max()  # integer index in df
    start_idx = end_idx - (window - 1)
    if start_idx < 0:
        raise ValueError(
            f"Not enough history in source series to build window={window} (need {window} points)."
        )
    return start_idx, end_idx


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
    Parameters
    ----------
    series: pandas.Series (index preserved)
    query_slice: slice or (start, end)
    window: length of sliding window; if None, uses len(query)
    method: distance method
    top_k: number of top matches to return (if threshold is None)
    threshold: if given, return all with distance <= threshold
    exclude_self: クエリ区間自身と重複する候補を除外

    Returns
    -------
    pandas.DataFrame with columns:
      - start_pos, end_pos: integer positions in series
      - start_index, end_index: original index labels
      - distance
      - subsequence: numpy array of the matched subsequence

    Notes
    -----
    - 欠損値は一旦削除してから処理（必要なら拡張）
    - クエリ区間の長さが window と異なる場合、クエリを中心クロップまたは平均値パディングして window 長に調整
    - DTW は計算コスト高いので注意
    - series の長さが window より短い場合は ValueError
    - query_slice は series のインデックスラベルを使えるが、存在しない場合は整数位置として解釈
    例: find_similar_subsequences(s, query_slice=(10,20), window=11, method='euclid', top_k=3)
    例: find_similar_subsequences(s, query_slice=slice('2023-01-01','2023-01-10'), window=None, method='dtw', threshold=5.0)
    例: find_similar_subsequences(s, query_slice=(5,15), window=11, method='cosine', exclude_self=False)
    例: find_similar_subsequences(s, query_slice=(0,9), window=10, method='euclid', znorm=False)
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


def find_similar_in_dataset(
    df: pd.DataFrame,
    target_time: datetime.datetime,
    window: int,
    product: str,
    parameter: str,
    product_scope: Optional[Union[str, List[str]]] = None,
    parameter_scope: Optional[Union[str, List[str]]] = None,
    top_k: int = 5,
    # pass-through params to find_similar_subsequences
    method: str = "euclid",
    znorm: bool = True,
    stride: int = 1,
    threshold: Optional[float] = None,
    dtw_window: Optional[int] = None,
    per_series_top_k: int = 3,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """
    Search dataset for subsequences similar to the window-ending-at(target_time)
    subsequence of (product, parameter).

    Parameters
    ----------
    df : pd.DataFrame
        must contain columns: ['login_date'(datetime), 'product', 'parameter', 'value']
    target_time : datetime.datetime
        the time point which is the end point of the query subsequence
    window : int
        how many latest points (including the point at/just-before target_time) to form query
    product : str
        source product name (where query is taken)
    parameter : str
        source parameter name
    product_scope : None | str | list[str]
        which products to search among. None => all products. If str, exact match.
        If list, use that list.
    parameter_scope : None | str | list[str]
        which parameters to search among. None => all parameters.
    top_k : int
        number of top matches overall to return
    per_series_top_k : int
        how many matches to ask find_similar_subsequences to return per candidate series
    Other args are forwarded to find_similar_subsequences.

    Returns
    -------
    pd.DataFrame (sorted by distance asc) with columns:
      ['product','parameter','start_pos','end_pos','start_time','end_time','distance','subsequence']
    """
    # --- validation & preprocessing ---
    df = check_cols_and_ensure_datetime(df)

    # filter source series
    src_mask = (df["product"] == product) & (df["parameter"] == parameter)
    src_df = df[src_mask].sort_values("login_date").reset_index(drop=True)

    if src_df.empty:
        raise ValueError(f"No data for source product={product}, parameter={parameter}")

    # find row index to use as end-of-window: last row with login_date <= target_time
    start_idx, end_idx = find_start_end_index(src_df, target_time, window)

    # extract query subsequence (numpy array)
    query_slice_df = src_df.loc[start_idx:end_idx].copy()
    query_vals = query_slice_df["value"].to_numpy()
    # if NaNs present, raise or drop depending on policy; here we require no NaN in query
    if np.isnan(query_vals).any():
        raise ValueError("Query subsequence contains NaN; please preprocess missing values.")

    query_len = len(query_vals)
    if query_len != window:
        # defensive, should not happen
        raise ValueError("Query length mismatch.")

    # prepare search scope lists
    all_products = sorted(df["product"].unique())
    all_parameters = sorted(df["parameter"].unique())

    def normalize_scope(x, all_values):
        if x is None:
            return all_values
        if isinstance(x, str):
            return [x]
        if isinstance(x, (list, tuple, set)):
            return [str(xx) for xx in x]
        raise ValueError("scope must be None, str, or list-like")

    product_candidates = normalize_scope(product_scope, all_products)
    parameter_candidates = normalize_scope(parameter_scope, all_parameters)

    # --- iterate candidate series ---
    results = []
    # to speed, we will groupby product/parameter
    grp = df.sort_values("login_date").groupby(["product", "parameter"])

    for (p, param), group in grp:
        # skip if not in search scope
        if (p not in product_candidates) or (param not in parameter_candidates):
            continue

        # we will compare query to this group's series
        series_values = group.sort_values("login_date")["value"].reset_index(drop=True)
        series_index = group.sort_values("login_date")["login_date"].reset_index(drop=True)

        # require at least window points to have any match
        if len(series_values) < 1:
            continue
        # construct combined series: candidate_series + query appended
        # Use integer index to make positions simple:
        combined_vals = np.concatenate([series_values.to_numpy(), query_vals])
        # create pd.Series with integer index 0..N-1
        combined_series = pd.Series(combined_vals)

        # query slice in combined series: start at len(series_values) and end at len(series_values)+query_len-1
        q_start_pos = len(series_values)
        q_end_pos = q_start_pos + query_len - 1

        # call the provided function
        try:
            matches_df = find_similar_subsequences(
                series=combined_series,
                query_slice=(q_start_pos, q_end_pos),
                window=query_len,
                stride=stride,
                method=method,
                znorm=znorm,
                top_k=per_series_top_k,
                threshold=threshold,
                dtw_window=dtw_window,
                exclude_self=exclude_self,
            )
        except Exception as e:
            # skip this series on error (e.g., DTW failure), but continue others
            # optionally you could raise; we choose to skip with continue
            # (you may log e in real system)
            continue

        # matches_df expected columns: start_pos, end_pos, start_index, end_index, distance, subsequence
        # we only want matches that are located in the original candidate series area (i.e., start_pos < q_start_pos)
        if matches_df is None or matches_df.empty:
            continue

        # iterate matches and map positions back to candidate's timestamps if possible
        for _, row in matches_df.iterrows():
            m_start = int(row["start_pos"])
            m_end = int(row["end_pos"])
            # ignore matches that are entirely within the appended query (should be excluded by exclude_self, but double-check)
            if m_start >= q_start_pos:
                continue
            # map start/end to timestamps (if within bounds)
            # ensure indices within candidate series length
            if m_end >= q_start_pos:
                # partial overlap with appended query -> ignore (usually happens only when overlap allowed)
                continue
            # convert start/end pos into timestamps from series_index (which is 0-based)
            try:
                start_time = pd.to_datetime(series_index.iloc[m_start])
                end_time = pd.to_datetime(series_index.iloc[m_end])
            except Exception:
                start_time = None
                end_time = None

            results.append(
                {
                    "product": p,
                    "parameter": param,
                    "start_pos": m_start,
                    "end_pos": m_end,
                    "start_time": start_time,
                    "end_time": end_time,
                    "distance": float(row["distance"]),
                    "subsequence": np.asarray(row["subsequence"], dtype=float),
                }
            )

    if len(results) == 0:
        # return empty DataFrame with expected columns
        cols = [
            "product",
            "parameter",
            "start_pos",
            "end_pos",
            "start_time",
            "end_time",
            "distance",
            "subsequence",
        ]
        return pd.DataFrame(columns=cols)

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("distance", ascending=True).reset_index(drop=True)

    # if exclude_self True, remove candidate equal to the source and overlapping the same time window (already excluded earlier normally)
    if exclude_self:
        mask_keep = ~(
            (res_df["product"] == product)
            & (res_df["parameter"] == parameter)
            & (res_df["end_time"] == pd.to_datetime(src_df.loc[end_idx, "login_date"]))
        )
        res_df = res_df[mask_keep].reset_index(drop=True)

    # limit to top_k
    if top_k is not None and len(res_df) > top_k:
        res_df = res_df.head(top_k).reset_index(drop=True)

    return res_df


def plot_similar_series(
    df: pd.DataFrame,
    target_time: datetime.datetime,
    window: int,
    product: str,
    parameter: str,
    product_scope: Optional[Union[str, list]] = None,
    parameter_scope: Optional[Union[str, list]] = None,
    top_k: int = 5,
    subplot_rows: int = 2,
    subplot_cols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    ax=None,
    stride: int = 5,
    # pass-through to find_similar_in_dataset (optional tuning)
    method: str = "euclid",
    znorm: bool = True,
    threshold: Optional[float] = None,
    dtw_window: Optional[int] = None,
    exclude_self: bool = True,
):
    """
    指定時刻を終点とする直近 window 長のクエリ系列と類似する系列を検索し、プロットする。

    Returns
    -------
    fig, ax
      matplotlib.figure.Figure, numpy.ndarray of Axes (flattened)
    """
    # 外部ユーティリティ（ユーザー環境で提供されている前提）
    # from libs.search import find_similar_in_dataset, find_start_end_index, check_cols_and_ensure_datetime

    # --- 前処理・検査 ---
    if figsize is None:
        # 軽い目安: 横4単位 × cols, 縦3単位 × rows
        figsize = (4 * subplot_cols, 3 * subplot_rows)

    max_plots = subplot_rows * subplot_cols
    # 1 plot is reserved for the target series, so available for results = max_plots - 1
    available_result_slots = max_plots - 1
    if available_result_slots < 0:
        raise ValueError("subplot_rows*subplot_cols must be at least 1")

    # cap top_k to available slots
    if top_k > available_result_slots:
        top_k_plot = available_result_slots
    else:
        top_k_plot = top_k

    # ensure df has necessary columns and datetime
    df_checked = check_cols_and_ensure_datetime(df.copy())

    # get target series df (only for the specified product/parameter)
    df_target = df_checked.loc[
        (df_checked["product"] == product) & (df_checked["parameter"] == parameter)
    ].sort_values("login_date")
    if df_target.empty:
        raise ValueError(f"No data for product={product}, parameter={parameter}")

    # find start/end indices in the target series for the window ending at target_time
    start_idx, end_idx = find_start_end_index(df_target, target_time, window)
    # extract the target subsequence values (as 1D array)
    target_vals = df_target.loc[start_idx:end_idx, "value"].reset_index(drop=True).to_numpy()

    # --- 検索 ---
    df_result = find_similar_in_dataset(
        df=df_checked,
        target_time=target_time,
        window=window,
        stride=stride,
        product=product,
        parameter=parameter,
        product_scope=product_scope,
        parameter_scope=parameter_scope,
        top_k=top_k,  # ask for top_k (function itself will cap)
        method=method,
        znorm=znorm,
        threshold=threshold,
        dtw_window=dtw_window,
        exclude_self=exclude_self,
    )

    # df_result の確認
    if df_result is None or df_result.empty:
        # プロットはターゲットのみ
        n_results = 0
    else:
        n_results = min(len(df_result), top_k_plot)

    # --- Figure/Axes の準備 ---
    if ax is None:
        fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)  # flatten
    else:
        # ax が渡された場合、いくつかの形を受け入れる
        if isinstance(ax, plt.Axes):
            axes = np.array([ax])
            fig = ax.figure
        else:
            # ax expected to be array-like of Axes
            axes = np.asarray(ax).reshape(-1)
            fig = axes[0].figure if len(axes) > 0 else plt.gcf()

        # 足りない場合はエラー
        if len(axes) < max_plots:
            raise ValueError(
                f"Provided ax has {len(axes)} axes but need at least {max_plots} (rows*cols)."
            )

    # --- 描画 ---
    # 0番目: ターゲット系列
    idx0 = 0
    axes[idx0].plot(np.arange(len(target_vals)), target_vals)
    # axes[idx0].set_title(f"{parameter}/{product} (end {pd.to_datetime(target_time)})")
    axes[idx0].set_title(f"{parameter}/{product}")
    axes[idx0].set_xlabel("Index (window)")
    axes[idx0].set_ylabel("Value")

    # 描画スタイル簡潔化: 同じY軸スケールを使いたければ後で合わせる
    # 検索結果を順に描画（距離昇順で並んでいる想定）
    if n_results > 0:
        # ensure df_result sorted by distance ascending
        df_result_sorted = (
            df_result.sort_values("distance", ascending=True)
            .reset_index(drop=True)
            .iloc[:n_results]
        )

        for i, (_, row) in enumerate(df_result_sorted.iterrows()):
            ax_idx = i + 1  # axes[1].. が最初の結果
            subseq = np.asarray(row["subsequence"], dtype=float)
            axes[ax_idx].plot(np.arange(len(subseq)), subseq)
            # タイトルに製品・パラメータ・距離・時刻を詰める
            title_time = ""
            if "start_time" in row and "end_time" in row:
                try:
                    st = pd.to_datetime(row["start_time"])
                    et = pd.to_datetime(row["end_time"])
                    # title_time = f"{st.strftime('%Y-%m-%d %H:%M')}~{et.strftime('%Y-%m-%d %H:%M')}"
                    title_time = ""
                except Exception:
                    title_time = ""
            axes[ax_idx].set_title(
                f"{row.get('parameter')}/{row.get('product')}/d={row.get('distance'):.2f} {title_time}"
            )
            axes[ax_idx].set_xlabel("Index (window)")
            axes[ax_idx].set_ylabel("Value")

    # unused axes: turn off
    total_needed = 1 + n_results
    for j in range(total_needed, max_plots):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig, axes
