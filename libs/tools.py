import datetime
import os
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from libs.search import plot_similar_series

CURDIR = os.path.dirname(os.path.abspath(__file__))


def save_similar_series_plot(
    filepath: str,
    target_time: Union[str, datetime.datetime],
    window: int,
    product: str,
    parameter: str,
) -> None:
    """
    Generate and save a plot of time-series subsequences similar to a target subsequence.

    Notes
    -----
    - The implementation expects `plot_similar_series` to be available in the same
      runtime (imported or defined in the module) and to accept the arguments used
      in the call below.
    - The CSV load path is relative to `CURDIR`: make sure `CURDIR` is defined and
      points to the module directory in your environment.
    - This function will overwrite `filepath` if a file already exists at that location.
    - `target_time` may be provided as a `datetime.datetime` or as a string accepted by
      `pandas.to_datetime` (e.g. "2025-08-01 12:31:00" or "2025-08-01").
    - The saved figure is closed before returning (so no interactive window is left open).

    Parameters
    ----------
    filepath : str
        Destination path (including filename and extension) to save the figure,
        e.g. "out/similar_plot.png". Parent directories must exist or an OSError will be raised.
    target_time : str | datetime.datetime
        The time point used as the end of the query window. If a string is provided,
        it will be parsed by pandas.to_datetime.
    window : int
        Number of most-recent points (including the point at-or-before `target_time`)
        used to form the query subsequence.
    product : str
        Product name where the target subsequence is taken from.
    parameter : str
        Parameter name where the target subsequence is taken from.

    Examples
    --------
    >>> save_similar_series_plot(
    ...     "out/plot1.png",
    ...     "2024-09-01 17:00:00",
    ...     window=10,
    ...     product="製品0",
    ...     parameter="FlowRate_L_min",
    ... )
    'out/plot1.png'

    >>> from datetime import datetime
    >>> save_similar_series_plot("out/plot2.png", datetime(2024,9,1,17,0), 10, "製品0", "FlowRate_L_min")
    'out/plot2.png'
    """
    # Parse target_time string to datetime if necessary
    if isinstance(target_time, str):
        try:
            target_time = pd.to_datetime(target_time)
        except Exception as e:
            raise ValueError(f"Unable to parse target_time string: {target_time}") from e

    # Load dataset (relative path from module's CURDIR)
    csv_path = os.path.join(CURDIR, "../data/trend/trend_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected dataset CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Call the plotting utility (must be available in scope)
    fig, ax = plot_similar_series(
        df=df,
        target_time=target_time,
        window=window,
        stride=1,
        product=product,
        parameter=parameter,
        product_scope=None,
        parameter_scope=parameter,
        top_k=5,
        method="dtw",
    )

    # Ensure layout and save
    fig.tight_layout()
    # create parent dir if desired behavior is to auto-create (commented out by default):
    # os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath)
    plt.close(fig)
