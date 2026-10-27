"""Reusable plotting helpers for TailDart notebooks."""

from __future__ import annotations

from math import ceil
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helpers.forecast_eval import DEFAULT_METRIC_COLUMNS

DEFAULT_METRIC_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
DEFAULT_MODEL_COLORS = {
     "small": "#9F35D7", "base": "#0F16D7", "large": "#DA2A2A",
    "supervised": "#4C72B0", "zero_shot": "#DD8452", "one_shot": "#55A859"
}

def _display_model_name(model: str) -> str:
    """Return the public model name used in figure text."""
    return model


def plot_metrics_by_asset_type(
    metrics_df: pd.DataFrame,
    metric_columns: Sequence[str] | None = None,
    *,
    asset_type_col: str = "asset_type",
    title: str | None = None,
) -> Figure:
    """Plot already-aggregated metrics by asset type.

    Args:
        metrics_df:
            Per-asset-type metrics table with shape ``(n_asset_types, n_columns)``.
            Expected columns are:
            - ``asset_type_col``: asset type label, e.g. ``Crypto``.
            - one numeric column for each metric in ``metric_columns`` or
              ``DEFAULT_METRIC_COLUMNS`` by default.
        metric_columns:
            Ordered metric column names to aggregate and plot. Shape ``(n_metrics,)``.
            If ``None``, metrics are taken from ``DEFAULT_METRIC_COLUMNS``, the
            shared source of truth defined in ``helpers.forecast_eval``.
        asset_type_col:
            Name of the asset type column in ``metrics_df``.
        title:
            Optional figure title.

    Returns:
        ``fig``, a Matplotlib figure containing one subplot per metric.

    Raises:
        ValueError: If ``metrics_df`` is empty or missing required columns.
    """
    if metrics_df.empty:
        raise ValueError("metrics_df is empty; compute metrics before plotting.")

    if asset_type_col not in metrics_df.columns:
        raise ValueError(f"metrics_df must include '{asset_type_col}'.")

    metrics = list(metric_columns or DEFAULT_METRIC_COLUMNS)
    missing_metrics = [column for column in metrics if column not in metrics_df.columns]
    if missing_metrics:
        missing = ", ".join(missing_metrics)
        raise ValueError(f"metrics_df is missing required metric columns: {missing}.")

    grouped = metrics_df[[asset_type_col, *metrics]].copy().sort_values(asset_type_col).reset_index(drop=True)

    n_metrics = len(metrics)
    ncols = min(3, n_metrics)
    nrows = ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    colors = [DEFAULT_METRIC_COLORS[idx % len(DEFAULT_METRIC_COLORS)] for idx in range(n_metrics)]
    for ax, metric, color in zip(axes_flat, metrics, colors):
        ax.bar(grouped[asset_type_col], grouped[metric], color=color)
        ax.set_title(metric.upper())
        ax.set_xlabel("Asset Type")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=45)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig


def plot_model_comparison_by_asset_type(
    metrics_df: pd.DataFrame,
    metric: str,
    *,
    model_col: str = "model",
    window_col: str = "window",
    asset_type_col: str = "asset_type",
    title: str | None = None,
) -> Figure:
    """Plot one metric across windows for each asset type, split by model."""
    if metrics_df.empty:
        raise ValueError("metrics_df is empty; compute metrics before plotting.")

    required_columns = [model_col, window_col, asset_type_col, metric]
    missing_columns = [column for column in required_columns if column not in metrics_df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"metrics_df is missing required columns: {missing}.")

    plot_df = metrics_df[required_columns].copy()
    plot_df = plot_df.sort_values([asset_type_col, model_col, window_col]).reset_index(drop=True)

    asset_types = sorted(plot_df[asset_type_col].dropna().astype(str).unique().tolist())
    models = sorted(plot_df[model_col].dropna().astype(str).unique().tolist())
    if not asset_types:
        raise ValueError("metrics_df must contain at least one asset type.")
    if not models:
        raise ValueError("metrics_df must contain at least one model.")

    n_assets = len(asset_types)
    ncols = min(4, n_assets)
    nrows = ceil(n_assets / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.75 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    cmap = plt.get_cmap("tab10")
    model_colors = {
        model: DEFAULT_MODEL_COLORS.get(model, cmap(idx % 10))
        for idx, model in enumerate(models)
    }

    legend_handles: list[Line2D] = []
    for model in models:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=model_colors[model],
                marker="o",
                linewidth=2,
                label=_display_model_name(model),
            )
        )

    for ax, asset_type in zip(axes_flat, asset_types):
        asset_df = plot_df[plot_df[asset_type_col].astype(str) == asset_type]
        for model in models:
            model_df = asset_df[asset_df[model_col].astype(str) == model].sort_values(window_col)
            if model_df.empty:
                continue
            ax.plot(
                model_df[window_col],
                model_df[metric],
                color=model_colors[model],
                marker="o",
                linewidth=2,
            )

        ax.set_title(asset_type)
        ax.set_xlabel("Window")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(sorted(asset_df[window_col].dropna().unique().tolist()))
        ax.grid(axis="y", alpha=0.25)

    for ax in axes_flat[n_assets:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, y=0.995)
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=min(len(legend_handles), 4),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
    else:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=min(len(legend_handles), 4),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_metrics_heatmap_by_asset_type(
    metrics_df: pd.DataFrame,
    metric_columns: Sequence[str] | None = DEFAULT_METRIC_COLUMNS,
    *,
    asset_type_col: str = "asset_type",
    title: str | None = None,
    annotate: bool = True,
    normalize: bool = True,
    invert_lower_is_better: bool = True,
) -> Figure:
    """Plot a single asset-type x metric heatmap from aggregated metrics.

    Color is based on per-metric normalized values so columns are comparable.
    By default, lower-is-better metrics are inverted so larger color values
    consistently indicate better performance.
    """
    if metrics_df.empty:
        raise ValueError("metrics_df is empty; compute metrics before plotting.")

    if asset_type_col not in metrics_df.columns:
        raise ValueError(f"metrics_df must include '{asset_type_col}'.")

    metrics = list(metric_columns or DEFAULT_METRIC_COLUMNS)
    missing_metrics = [column for column in metrics if column not in metrics_df.columns]
    if missing_metrics:
        missing = ", ".join(missing_metrics)
        raise ValueError(f"metrics_df is missing required metric columns: {missing}.")

    grouped = (
        metrics_df[[asset_type_col, *metrics]]
        .copy()
        .sort_values(asset_type_col)
        .reset_index(drop=True)
    )
    grouped[asset_type_col] = grouped[asset_type_col].astype(str)

    raw_values = grouped[metrics].astype(float)
    display_values = raw_values.copy()
    score_values = raw_values.copy()

    if normalize:
        for metric in metrics:
            col = raw_values[metric]
            min_val = float(col.min())
            max_val = float(col.max())
            if np.isclose(min_val, max_val):
                score_values[metric] = 1.0
            else:
                score_values[metric] = (col - min_val) / (max_val - min_val)
    else:
        score_values = raw_values.copy()

    if invert_lower_is_better:
        lower_is_better = set(DEFAULT_METRIC_COLUMNS)
        for metric in metrics:
            if metric.lower() in lower_is_better:
                score_values[metric] = 1.0 - score_values[metric]

    matrix = score_values.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(1.4 * len(metrics) + 4, 0.65 * len(grouped) + 2.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized score (higher is better)")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([metric.upper() for metric in metrics], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(grouped)))
    ax.set_yticklabels(grouped[asset_type_col].tolist())
    ax.set_xlabel("Metric")
    ax.set_ylabel("Asset Type")

    if annotate:
        for row_idx in range(matrix.shape[0]):
            for col_idx, metric in enumerate(metrics):
                score = matrix[row_idx, col_idx]
                raw = float(display_values.iloc[row_idx, col_idx])
                text_color = "white" if score >= 0.6 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{score:.2f}\n({raw:.4g})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_window_radar_by_asset_type(
    metrics_df: pd.DataFrame,
    metric_columns: Sequence[str] | None = DEFAULT_METRIC_COLUMNS,
    *,
    asset_type_col: str = "asset_type",
    window_col: str = "window",
    title: str | None = None,
):
    if metrics_df.empty:
        raise ValueError("metrics_df is empty.")

    metrics = list(metric_columns)

    required = [asset_type_col, window_col, *metrics]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"metrics_df missing required columns: {missing}")

    df = metrics_df.copy()
    asset_types = sorted(df[asset_type_col].astype(str).unique())
    windows = sorted(df[window_col].unique())
    if not asset_types:
        raise ValueError("metrics_df must contain at least one asset type.")
    if not windows:
        raise ValueError("metrics_df must contain at least one window.")

    normalized_df = df.copy()

    for metric in metrics:
        values = df[metric].astype(float)
        min_val = values.min()
        max_val = values.max()
        if np.isclose(min_val, max_val):
            normalized_df[metric] = 1.0
        else:
            normalized_df[metric] = 1.0 - (values - min_val) / (max_val - min_val)

    n_assets = len(asset_types)
    ncols = min(4, n_assets)
    nrows = int(np.ceil(n_assets / ncols))
    theta_labels = [metric.upper() for metric in metrics]

    colors = DEFAULT_METRIC_COLORS
    window_colors = {
        w: colors[i % len(colors)]
        for i, w in enumerate(windows)
    }

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4.5 * nrows),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )
    axes_flat = axes.flatten()
    legend_handles = [
        Line2D([0], [0], color=window_colors[window], linewidth=2, label=f"window={window}")
        for window in windows
    ]

    for ax, asset_type in zip(axes_flat, asset_types):
        asset_df = normalized_df[normalized_df[asset_type_col].astype(str) == asset_type]
        asset_df = asset_df.sort_values(window_col).reset_index(drop=True)

        for _, row in asset_df.iterrows():
            values = [float(row[metric]) for metric in metrics]
            values += values[:1]
            window = row[window_col]
            color = window_colors.get(window, "#333333")
            ax.plot(angles, values, color=color, linewidth=2)
            ax.fill(angles, values, color=color, alpha=0.08)

        ax.set_title(asset_type, pad=16)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(theta_labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes_flat[n_assets:]:
        ax.set_visible(False)

    fig.legend(handles=legend_handles, loc="upper center", ncol=min(len(legend_handles), 3), frameon=False)
    fig.text(
        0.5,
        0.03,
        "Note: values are globally normalized and inverted; farther from center = better.",
        ha="center",
        va="center",
        fontsize=10,
    )
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    else:
        fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    return fig

__all__ = [
    "plot_metrics_by_asset_type",
    "plot_metrics_heatmap_by_asset_type",
    "plot_model_comparison_by_asset_type",
    "plot_window_radar_by_asset_type",
]
