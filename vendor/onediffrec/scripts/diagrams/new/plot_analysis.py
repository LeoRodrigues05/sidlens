#!/usr/bin/env python3
"""Create the collision-rate analysis tables, figures, and narrative report."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from functools import lru_cache
from itertools import permutations
from pathlib import Path

# Keep Matplotlib's cache in a writable, non-project location on compute nodes.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "onediffrec-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy import stats

from parse_workbook import (
    DEFAULT_WORKBOOK,
    METRICS,
    TOKENIZER_ORDER,
    build_tables,
)
from parse_research_notes import DEFAULT_PDF, build_office_table


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "diagrams" / "new"

# Okabe-Ito-inspired colors: readable in color-vision-deficiency simulations.
TOKENIZER_COLORS = {
    "RQ-VAE": "#0072B2",
    "RQ-KMeans": "#D55E00",
    "MQ": "#009E73",
}
FAMILY_COLORS = {
    "Autoregressive": "#3B6FB6",
    "Diffusion-family": "#CC6677",
    "Paired-config mean": "#228833",
}
FAMILY_MARKERS = {"Autoregressive": "o", "Diffusion-family": "^"}
TOKENIZER_MARKERS = {"RQ-VAE": "o", "RQ-KMeans": "s", "MQ": "^"}
CODEBOOK_MARKERS = {128: "o", 256: "s", 512: "^"}
DEPTH_SIZES = {3: 45, 4: 78, 5: 118}

SURFACE = "#FFFFFF"
INK = "#171717"
INK_2 = "#4B4B48"
MUTED = "#777771"
GRID = "#D9D9D9"
MATRIX_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "onediffrec_blue", ["#F7FBFF", "#9ECAE1", "#3182BD", "#08519C"]
)
MATRIX_DIVERGING = LinearSegmentedColormap.from_list(
    "onediffrec_orange_blue", ["#D95F02", "#F7F7F7", "#0072B2"]
)

# The original suite used an 8.5 pt legend.  Keep the requested increase
# explicit and testable instead of scattering per-figure font overrides.
LEGEND_BASE_FONTSIZE = 8.5
LEGEND_SCALE = 1.8
LEGEND_FONTSIZE = LEGEND_BASE_FONTSIZE * LEGEND_SCALE
ANNOTATION_FONTSIZE = 10.5
VALUE_LABEL_FONTSIZE = 9.5
PANEL_LABELS = "abcdefghijklmnopqrstuvwxyz"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.9,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.labelsize": 12,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.5,
            "grid.linestyle": ":",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.frameon": False,
            "legend.fontsize": LEGEND_FONTSIZE,
            "legend.title_fontsize": LEGEND_FONTSIZE,
            "legend.handlelength": 1.7,
            "legend.handletextpad": 0.55,
            "legend.columnspacing": 1.1,
            "legend.markerscale": 1.25,
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Keep figure-level titles in the paper caption, not inside the image."""

    # Retain the call site API so individual figures still document their
    # intended caption text in code, while producing title-free image assets.
    _ = (fig, title, subtitle)


def add_footer(fig: plt.Figure, text: str, y: float = 0.018) -> None:
    fig.text(
        0.02,
        y,
        text,
        ha="left",
        va="bottom",
        fontsize=VALUE_LABEL_FONTSIZE,
        color=MUTED,
    )


def add_panel_labels(axes: object, x: float = -0.13, y: float = 1.06) -> None:
    """Apply conventional (a), (b), ... panel labels to an axes collection."""

    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    for index, ax in enumerate(flat_axes):
        ax.text(
            x,
            y,
            f"({PANEL_LABELS[index]})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            weight="bold",
            color=INK,
            clip_on=False,
        )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor=SURFACE,
            metadata={"Creator": "OneDiffRec Python/Matplotlib figure pipeline"},
        )
    plt.close(fig)
    print(f"  {stem}.png / .pdf")


def _p_text(value: float) -> str:
    if value < 0.001:
        return "p < .001"
    return f"p = {value:.3f}".replace("0.", ".")


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    result = stats.spearmanr(x.astype(float), y.astype(float))
    return float(result.statistic), float(result.pvalue)


@lru_cache(maxsize=4)
def _rank_permutations(sorted_ranks: tuple[float, ...]) -> np.ndarray:
    """Return every labeled permutation of a small rank vector.

    Within-tokenizer tests have n=9, so the complete 9! reference distribution
    is both feasible and preferable to the large-sample p-value from spearmanr.
    Repeated rank values intentionally remain labeled permutations, matching a
    randomization test over the nine configuration cells.
    """

    return np.asarray(list(permutations(sorted_ranks)), dtype=np.float32)


def _exact_spearman_p(x: pd.Series, y: pd.Series) -> float:
    x_ranks = stats.rankdata(x.astype(float))
    y_ranks = stats.rankdata(y.astype(float))
    observed = float(np.corrcoef(x_ranks, y_ranks)[0, 1])
    all_y = _rank_permutations(tuple(sorted(float(value) for value in y_ranks)))
    x_centered = x_ranks - x_ranks.mean()
    denominator = np.sqrt(
        np.sum(x_centered**2) * np.sum((y_ranks - y_ranks.mean()) ** 2)
    )
    null_rhos = (all_y @ x_centered) / denominator
    return float(np.mean(np.abs(null_rhos) >= abs(observed) - 1e-10))


def configuration_means(core_runs: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "tokenizer",
        "depth",
        "codebook",
        "collision_pct",
        "collision_excess",
        "collision_source_status",
    ]
    means = (
        core_runs.groupby(keys, observed=True, as_index=False)[METRICS]
        .mean()
        .sort_values(["tokenizer", "codebook", "depth"])
        .reset_index(drop=True)
    )
    means["n_paradigms"] = 2
    return means


def paradigm_gaps(core_runs: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "tokenizer",
        "depth",
        "codebook",
        "collision_pct",
        "collision_source_status",
    ]
    base = core_runs[keys].drop_duplicates().copy()
    for metric in METRICS:
        wide = core_runs.pivot(
            index=keys, columns="family", values=metric
        ).reset_index()
        wide[f"{metric}_gap"] = (
            wide["Diffusion-family"] - wide["Autoregressive"]
        )
        base = base.merge(wide[keys + [f"{metric}_gap"]], on=keys, validate="one_to_one")
    return base.sort_values(["tokenizer", "codebook", "depth"]).reset_index(drop=True)


def factorial_effects(core_runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize balanced main effects without presenting ranges as uncertainty."""

    records: list[dict[str, object]] = []
    factor_levels: list[tuple[str, list[object]]] = [
        ("tokenizer", list(TOKENIZER_ORDER)),
        ("depth", [3, 4, 5]),
        ("codebook", [128, 256, 512]),
    ]
    for metric in ("HR@10", "NDCG@10"):
        for factor, levels in factor_levels:
            for level_index, level in enumerate(levels):
                for family in ("Autoregressive", "Diffusion-family"):
                    values = core_runs[
                        (core_runs[factor] == level) & (core_runs["family"] == family)
                    ][metric].astype(float)
                    if len(values) != 9:
                        raise ValueError(
                            f"Expected 9 balanced cells for {metric}/{factor}/{level}/{family}; "
                            f"found {len(values)}"
                        )
                    records.append(
                        {
                            "metric": metric,
                            "factor": factor,
                            "level": level,
                            "level_index": level_index,
                            "family": family,
                            "n_design_cells": len(values),
                            "mean": values.mean(),
                            "minimum": values.min(),
                            "maximum": values.max(),
                        }
                    )
    return pd.DataFrame.from_records(records)


def cross_dataset_performance_tables(
    core_runs: pd.DataFrame, office_runs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align the complete codebook-256 performance grids across datasets."""

    columns = [
        "dataset",
        "tokenizer",
        "depth",
        "codebook",
        "model",
        "family",
        *METRICS,
    ]
    industrial = core_runs[core_runs["codebook"] == 256][columns].copy()
    office = office_runs[columns].copy()
    combined = pd.concat([industrial, office], ignore_index=True)
    key = ["dataset", "tokenizer", "depth", "codebook", "family"]
    if len(combined) != 36 or combined.duplicated(key).any():
        raise ValueError("Cross-dataset codebook-256 grid must contain 36 unique rows")

    gap_base = combined[["dataset", "tokenizer", "depth", "codebook"]].drop_duplicates()
    for metric in METRICS:
        wide = combined.pivot(
            index=["dataset", "tokenizer", "depth", "codebook"],
            columns="family",
            values=metric,
        ).reset_index()
        wide[f"{metric}_gap"] = wide["Diffusion-family"] - wide["Autoregressive"]
        gap_base = gap_base.merge(
            wide[["dataset", "tokenizer", "depth", "codebook", f"{metric}_gap"]],
            on=["dataset", "tokenizer", "depth", "codebook"],
            validate="one_to_one",
        )

    agreement = industrial.merge(
        office,
        on=["tokenizer", "depth", "codebook", "family"],
        suffixes=("_Industrial", "_Office"),
        validate="one_to_one",
    )
    return (
        combined.sort_values(["dataset", "tokenizer", "depth", "family"]).reset_index(
            drop=True
        ),
        gap_base.sort_values(["dataset", "tokenizer", "depth"]).reset_index(drop=True),
        agreement.sort_values(["tokenizer", "depth", "family"]).reset_index(drop=True),
    )


def association_tables(
    core_runs: pd.DataFrame, config_means: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = {
        "Autoregressive": core_runs[core_runs["family"] == "Autoregressive"],
        "Diffusion-family": core_runs[core_runs["family"] == "Diffusion-family"],
        "Paired-config mean": config_means,
    }
    records: list[dict[str, object]] = []
    for series_name, frame in series.items():
        for sensitivity, filtered in (
            ("All codebooks", frame),
            ("Exclude codebook 256", frame[frame["codebook"] != 256]),
        ):
            for metric in METRICS:
                rho, p_value = _spearman(filtered["collision_pct"], filtered[metric])
                records.append(
                    {
                        "series": series_name,
                        "sensitivity": sensitivity,
                        "metric": metric,
                        "n_configurations": len(filtered),
                        "spearman_rho": rho,
                        "p_value": p_value,
                    }
                )
    associations = pd.DataFrame.from_records(records)

    within_records: list[dict[str, object]] = []
    for tokenizer, frame in config_means.groupby("tokenizer", observed=True):
        for metric in METRICS:
            rho, _ = _spearman(frame["collision_pct"], frame[metric])
            p_value = _exact_spearman_p(frame["collision_pct"], frame[metric])
            within_records.append(
                {
                    "tokenizer": tokenizer,
                    "metric": metric,
                    "n_configurations": len(frame),
                    "collision_min_pct": frame["collision_pct"].min(),
                    "collision_max_pct": frame["collision_pct"].max(),
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "p_value_method": "exhaustive permutation over design cells",
                }
            )
    within = pd.DataFrame.from_records(within_records)
    return associations, within


def cross_dataset_table(collisions: pd.DataFrame) -> pd.DataFrame:
    wide = collisions.pivot(
        index=["tokenizer", "depth", "codebook"],
        columns="dataset",
        values="collision_pct",
    ).reset_index()
    wide.columns.name = None
    wide["office_to_industrial_ratio"] = wide["Office"] / wide["Industrial"]
    return wide.sort_values(["tokenizer", "codebook", "depth"]).reset_index(drop=True)


def build_derived_tables(
    tables: dict[str, pd.DataFrame], office_runs: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    core_runs = tables["core_runs"].copy()
    means = configuration_means(core_runs)
    gaps = paradigm_gaps(core_runs)
    associations, within = association_tables(core_runs, means)
    cross = cross_dataset_table(tables["collisions"])
    backbone = tables["runs"][
        (tables["runs"]["codebook"] == 256) & (tables["runs"]["depth"] == 3)
    ].copy()

    gap_summary = (
        gaps.groupby("depth", as_index=False)
        .agg(
            n_configurations=("tokenizer", "size"),
            HR10_mean_gap=("HR@10_gap", "mean"),
            HR10_median_gap=("HR@10_gap", "median"),
            NDCG10_mean_gap=("NDCG@10_gap", "mean"),
            NDCG10_median_gap=("NDCG@10_gap", "median"),
        )
        .sort_values("depth")
    )
    derived = {
        "configuration_means": means,
        "paradigm_gaps": gaps,
        "association_summary": associations,
        "within_tokenizer_associations": within,
        "cross_dataset_collisions": cross,
        "model_training_ablation_3d_codebook256": backbone,
        "paradigm_gap_summary": gap_summary,
        "factorial_effects": factorial_effects(core_runs),
    }
    if office_runs is not None:
        combined, cross_gaps, agreement = cross_dataset_performance_tables(
            core_runs, office_runs
        )
        derived.update(
            {
                "cross_dataset_performance_codebook256": combined,
                "cross_dataset_paradigm_gaps_codebook256": cross_gaps,
                "cross_dataset_performance_agreement_codebook256": agreement,
            }
        )
    return derived


def write_derived_tables(derived: dict[str, pd.DataFrame], output_dir: Path) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, table in derived.items():
        export = table.copy()
        float_columns = export.select_dtypes(include=["floating"]).columns
        export[float_columns] = export[float_columns].round(8)
        export.to_csv(data_dir / f"{name}.csv", index=False)


def fig01_collision_design_space(
    collisions: pd.DataFrame, output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.8), sharex=True, sharey=True)
    datasets = ["Industrial", "Office"]
    codebooks = [128, 256, 512]

    for row, dataset in enumerate(datasets):
        for col, codebook in enumerate(codebooks):
            ax = axes[row, col]
            panel = collisions[
                (collisions["dataset"] == dataset)
                & (collisions["codebook"] == codebook)
            ]
            is_previous = dataset == "Office" or codebook == 256

            for tokenizer in TOKENIZER_ORDER:
                frame = panel[panel["tokenizer"] == tokenizer].sort_values("depth")
                ax.plot(
                    frame["depth"],
                    frame["collision_pct"],
                    color=TOKENIZER_COLORS[tokenizer],
                    marker="o",
                    markersize=6.5,
                    linewidth=2.2,
                    label=tokenizer,
                )
            ax.set_xlim(2.78, 5.22)
            ax.set_ylim(0, 60)
            ax.set_xticks([3, 4, 5])
            ax.set_yticks([0, 20, 40, 60])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
            ax.set_title(f"Codebook {codebook}" + ("  †" if is_previous else ""))
            if row == 1:
                ax.set_xlabel("SID depth (digits)")
            if col == 0:
                ax.set_ylabel(f"{dataset}\ncollision rate")
            despine(ax)

    add_panel_labels(axes, x=0.015, y=0.96)

    add_header(
        fig,
        "Collision rate across SID design configurations",
        "Deeper SIDs and larger codebooks usually reduce excess SID assignments; RQ-VAE stays near the low-collision end.",
    )
    handles = [
        Line2D([0], [0], color=TOKENIZER_COLORS[tokenizer], marker="o", lw=2, label=tokenizer)
        for tokenizer in TOKENIZER_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3)
    add_footer(
        fig,
        "Collision rate follows the repository definition (N − unique full SIDs) / N. † The workbook marks Industrial codebook-256 and all Office values as old/previous runs.",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.14, wspace=0.22, hspace=0.30)
    save_figure(fig, output_dir, "fig01_collision_design_space")


def fig02_cross_dataset_collision(cross: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    bounds = (0.27, 70)
    diagonal = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), 200)
    ax.plot(diagonal, diagonal, color=INK_2, linestyle="--", linewidth=1.2, zorder=1)

    for row in cross.itertuples(index=False):
        face = TOKENIZER_COLORS[str(row.tokenizer)] if row.codebook != 256 else SURFACE
        ax.scatter(
            row.Industrial,
            row.Office,
            s=DEPTH_SIZES[int(row.depth)],
            marker=CODEBOOK_MARKERS[int(row.codebook)],
            facecolor=face,
            edgecolor=TOKENIZER_COLORS[str(row.tokenizer)],
            linewidth=1.6,
            alpha=0.92,
            zorder=3,
        )

    rho, p_value = _spearman(cross["Industrial"], cross["Office"])
    ax.text(
        0.035,
        0.96,
        f"Spearman ρ = {rho:.3f}\n{_p_text(p_value)}  ·  n = {len(cross)} configs",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": GRID},
    )

    outlier = cross[
        (cross["tokenizer"] == "RQ-KMeans")
        & (cross["depth"] == 5)
        & (cross["codebook"] == 512)
    ].iloc[0]
    ax.annotate(
        f"RQ-KMeans · 5d × 512\n{outlier.office_to_industrial_ratio:.1f}× higher in Office",
        (outlier.Industrial, outlier.Office),
        xytext=(1.15, 12.5),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": INK_2, "lw": 0.9},
        fontsize=ANNOTATION_FONTSIZE,
        color=INK_2,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect("equal", adjustable="box")
    ticks = [0.3, 1, 3, 10, 30, 60]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    formatter = FuncFormatter(lambda value, _: f"{value:g}%")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("Industrial collision rate")
    ax.set_ylabel("Office collision rate")
    despine(ax)

    tokenizer_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=TOKENIZER_COLORS[tokenizer],
            markeredgecolor=TOKENIZER_COLORS[tokenizer],
            label=tokenizer,
        )
        for tokenizer in TOKENIZER_ORDER
    ]
    codebook_handles = [
        Line2D(
            [0],
            [0],
            marker=CODEBOOK_MARKERS[codebook],
            linestyle="",
            markerfacecolor="white" if codebook == 256 else "#888888",
            markeredgecolor="#555555",
            markersize=7,
            label=f"codebook {codebook}" + (" †" if codebook == 256 else ""),
        )
        for codebook in (128, 256, 512)
    ]
    depth_handles = [
        plt.scatter([], [], s=DEPTH_SIZES[depth], color="#777777", label=f"{depth} digits")
        for depth in (3, 4, 5)
    ]
    legend_one = ax.legend(handles=tokenizer_handles, title="Tokenizer", loc="lower right")
    ax.add_artist(legend_one)
    ax.legend(
        handles=codebook_handles + depth_handles,
        title="Design",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    add_header(
        fig,
        "Cross-dataset collision-rate agreement",
        "Each point is the same tokenizer × depth × codebook configuration; the dashed line marks equal rates.",
    )
    add_footer(
        fig,
        "All Office values are marked as previous runs in the workbook. Hollow squares are the separately flagged Industrial codebook-256 values.",
    )
    fig.subplots_adjust(left=0.12, right=0.77, top=0.97, bottom=0.13)
    save_figure(fig, output_dir, "fig02_cross_dataset_collision")


def fig03_collision_vs_reported_metrics(
    core_runs: pd.DataFrame, means: pd.DataFrame, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.1))

    for ax, metric in zip(axes, ("HR@10", "NDCG@10")):
        for _, frame in core_runs.groupby(
            ["tokenizer", "depth", "codebook"], observed=True
        ):
            tokenizer = str(frame["tokenizer"].iloc[0])
            x_value = float(frame["collision_pct"].iloc[0])
            values = frame[metric].sort_values().to_numpy()
            ax.plot(
                [x_value, x_value],
                values,
                color=TOKENIZER_COLORS[tokenizer],
                alpha=0.27,
                linewidth=1.2,
                zorder=1,
            )

        for tokenizer in TOKENIZER_ORDER:
            for family, marker in FAMILY_MARKERS.items():
                frame = core_runs[
                    (core_runs["tokenizer"] == tokenizer)
                    & (core_runs["family"] == family)
                ]
                current = frame[frame["codebook"] != 256]
                previous = frame[frame["codebook"] == 256]
                ax.scatter(
                    current["collision_pct"],
                    current[metric],
                    s=54,
                    marker=marker,
                    facecolor=TOKENIZER_COLORS[tokenizer],
                    edgecolor="white",
                    linewidth=0.7,
                    alpha=0.9,
                    zorder=3,
                )
                ax.scatter(
                    previous["collision_pct"],
                    previous[metric],
                    s=54,
                    marker=marker,
                    facecolor=SURFACE,
                    edgecolor=TOKENIZER_COLORS[tokenizer],
                    linewidth=1.4,
                    zorder=3,
                )

        ax.scatter(
            means["collision_pct"],
            means[metric],
            s=42,
            marker="_",
            color=INK,
            linewidth=1.1,
            zorder=4,
        )
        log_x = np.log10(means["collision_pct"].to_numpy())
        slope, intercept, _, _ = stats.theilslopes(means[metric].to_numpy(), log_x)
        guide_x = np.logspace(np.log10(0.32), np.log10(45), 200)
        ax.plot(
            guide_x,
            intercept + slope * np.log10(guide_x),
            color=INK_2,
            linestyle="--",
            linewidth=1.4,
            zorder=2,
        )

        rho, p_value = _spearman(means["collision_pct"], means[metric])
        sensitivity = means[means["codebook"] != 256]
        rho_s, _ = _spearman(sensitivity["collision_pct"], sensitivity[metric])
        ax.text(
            0.03,
            0.97,
            f"27 config means: ρ = {rho:.3f}\nexclude 256: ρ = {rho_s:.3f}\n{_p_text(p_value)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONTSIZE,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": GRID},
        )
        ax.set_xscale("log")
        ax.set_xlim(0.28, 50)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
        ax.set_xlabel("Industrial collision rate (log scale)")
        ax.set_ylabel(f"Reported {metric}")
        ax.set_title(metric)
        despine(ax)

    tokenizer_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", color=TOKENIZER_COLORS[t], label=t
        )
        for t in TOKENIZER_ORDER
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            color="#555555",
            label=family,
        )
        for family, marker in FAMILY_MARKERS.items()
    ]
    artifact_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor=SURFACE,
        markeredgecolor="#555555",
        label="hollow marker = codebook 256 †",
    )
    mean_handle = Line2D([0], [0], marker="_", color=INK, linestyle="", label="paired mean")
    fig.legend(
        handles=tokenizer_handles + family_handles + [mean_handle, artifact_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
    )
    add_panel_labels(axes)
    add_header(
        fig,
        "Collision rate and ranking performance",
        "Vertical segments pair the two generator paradigms at the same nominal SID configuration; statistics use their mean once per configuration.",
    )
    add_footer(
        fig,
        "Dashed line: Theil–Sen descriptive guide on log10(collision). Association is not a causal effect; tokenizer, depth, and codebook co-vary. † old collision artifact.",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.76, bottom=0.16, wspace=0.25)
    save_figure(fig, output_dir, "fig03_collision_vs_reported_metrics")


def fig04_within_tokenizer_association(
    means: pd.DataFrame, within: pd.DataFrame, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.4), sharey=True)

    for ax, tokenizer in zip(axes, TOKENIZER_ORDER):
        frame = means[means["tokenizer"] == tokenizer].copy()
        color = TOKENIZER_COLORS[tokenizer]
        for codebook in (128, 256, 512):
            path = frame[frame["codebook"] == codebook].sort_values("depth")
            ax.plot(
                path["collision_pct"],
                path["HR@10"],
                color=color,
                alpha=0.32,
                linewidth=1.3,
            )
            for _, point in path.iterrows():
                face = SURFACE if codebook == 256 else color
                ax.scatter(
                    point["collision_pct"],
                    point["HR@10"],
                    s=DEPTH_SIZES[int(point["depth"])],
                    marker=CODEBOOK_MARKERS[codebook],
                    facecolor=face,
                    edgecolor=color,
                    linewidth=1.4,
                    zorder=3,
                )
        stat_row = within[
            (within["tokenizer"] == tokenizer) & (within["metric"] == "HR@10")
        ].iloc[0]
        ax.text(
            0.04,
            0.96,
            f"ρ = {stat_row.spearman_rho:+.3f}\npermutation {_p_text(stat_row.p_value)}\nrange {stat_row.collision_min_pct:.2f}–{stat_row.collision_max_pct:.2f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONTSIZE,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID},
        )
        span = frame["collision_pct"].max() - frame["collision_pct"].min()
        padding = max(span * 0.10, 0.05)
        low = max(0.0, frame["collision_pct"].min() - padding)
        high = frame["collision_pct"].max() + padding
        ax.set_xlim(low, high)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
        ax.set_title(tokenizer, color=color)
        ax.set_xlabel("Collision rate (panel-specific linear scale)")
        despine(ax)

    axes[0].set_ylabel("Mean reported HR@10 across paired paradigms")
    handles = [
        Line2D(
            [0],
            [0],
            marker=CODEBOOK_MARKERS[codebook],
            linestyle="",
            markerfacecolor="white" if codebook == 256 else "#777777",
            markeredgecolor="#555555",
            label=f"codebook {codebook}" + (" †" if codebook == 256 else ""),
        )
        for codebook in (128, 256, 512)
    ]
    depth_handles = [
        plt.scatter([], [], s=DEPTH_SIZES[depth], color="#777777", label=f"{depth} digits")
        for depth in (3, 4, 5)
    ]
    fig.legend(
        handles=handles + depth_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )
    add_panel_labels(axes)
    add_header(
        fig,
        "Within-tokenizer collision–performance association",
        "RQ-KMeans and MQ retain a positive within-tokenizer relationship; RQ-VAE has too little collision spread to show the same pattern.",
    )
    add_footer(
        fig,
        "Points are 27 SID configurations averaged across the two paradigms. Lines connect depths within a codebook; x-axis ranges differ by panel. † old collision artifact.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.77, bottom=0.18, wspace=0.18)
    save_figure(fig, output_dir, "fig04_within_tokenizer_association")


def fig05_metric_robustness(
    associations: pd.DataFrame, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    metric_order = METRICS[::-1]
    series_order = ["Autoregressive", "Diffusion-family", "Paired-config mean"]
    offsets = {"Autoregressive": -0.20, "Diffusion-family": 0.0, "Paired-config mean": 0.20}

    for metric_index, metric in enumerate(metric_order):
        for series_name in series_order:
            frame = associations[
                (associations["metric"] == metric)
                & (associations["series"] == series_name)
            ]
            full = frame[frame["sensitivity"] == "All codebooks"].iloc[0]
            sensitivity = frame[frame["sensitivity"] == "Exclude codebook 256"].iloc[0]
            y = metric_index + offsets[series_name]
            color = FAMILY_COLORS[series_name]
            ax.plot(
                [sensitivity.spearman_rho, full.spearman_rho],
                [y, y],
                color=color,
                alpha=0.38,
                linewidth=1.4,
            )
            ax.scatter(
                sensitivity.spearman_rho,
                y,
                s=48,
                facecolor=SURFACE,
                edgecolor=color,
                linewidth=1.4,
                zorder=3,
            )
            ax.scatter(full.spearman_rho, y, s=48, color=color, zorder=4)
            if series_name == "Paired-config mean":
                ax.text(
                    full.spearman_rho + 0.012,
                    y,
                    f"{full.spearman_rho:.2f}",
                    va="center",
                    fontsize=VALUE_LABEL_FONTSIZE,
                    color=color,
                )

    ax.set_yticks(range(len(metric_order)))
    ax.set_yticklabels(metric_order)
    ax.set_xlim(0.79, 0.90)
    ax.set_xticks(np.arange(0.80, 0.901, 0.02))
    ax.set_xlabel("Spearman correlation with Industrial collision rate")
    ax.set_ylabel("Reported metric")
    despine(ax)

    series_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=FAMILY_COLORS[name],
            label=name,
        )
        for name in series_order
    ]
    fill_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#666666", label="all codebooks"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=SURFACE,
            markeredgecolor="#666666",
            label="exclude codebook 256",
        ),
    ]
    fig.legend(
        handles=series_handles + fill_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )
    add_header(
        fig,
        "Sensitivity of collision–metric associations",
        "Solid markers use all 27 configurations; hollow markers remove the workbook's flagged codebook-256 collision values (n = 18).",
    )
    add_footer(
        fig,
        "Correlations are descriptive across configuration cells, not uncertainty across training seeds. The six metrics are highly redundant, so HR@10/NDCG@10 are sufficient headline views.",
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.76, bottom=0.17)
    save_figure(fig, output_dir, "fig05_metric_robustness")


def fig06_paradigm_gap_by_depth(gaps: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.6))
    tokenizer_offsets = {"RQ-VAE": -0.13, "RQ-KMeans": 0.0, "MQ": 0.13}
    codebook_offsets = {128: -0.035, 256: 0.0, 512: 0.035}

    for ax, metric in zip(axes, ("HR@10", "NDCG@10")):
        gap_col = f"{metric}_gap"
        for (tokenizer, codebook), path in gaps.groupby(
            ["tokenizer", "codebook"], observed=True
        ):
            path = path.sort_values("depth")
            offset = tokenizer_offsets[str(tokenizer)] + codebook_offsets[int(codebook)]
            x = path["depth"].to_numpy(dtype=float) + offset
            ax.plot(
                x,
                path[gap_col],
                color=TOKENIZER_COLORS[str(tokenizer)],
                alpha=0.25,
                linewidth=1.0,
            )
            ax.scatter(
                x,
                path[gap_col],
                s=52,
                marker=CODEBOOK_MARKERS[int(codebook)],
                facecolor=(
                    SURFACE if int(codebook) == 256 else TOKENIZER_COLORS[str(tokenizer)]
                ),
                edgecolor=TOKENIZER_COLORS[str(tokenizer)],
                linewidth=1.2,
                alpha=0.9,
                zorder=3,
            )

        mean_by_depth = gaps.groupby("depth")[gap_col].mean()
        ax.plot(
            mean_by_depth.index,
            mean_by_depth.values,
            color=INK,
            linewidth=2.6,
            marker="D",
            markersize=6.5,
            zorder=4,
        )
        for depth, value in mean_by_depth.items():
            ax.annotate(
                f"mean {value:+.2f}",
                (depth, value),
                xytext=(0, 9 if value >= 0 else -15),
                textcoords="offset points",
                ha="center",
                fontsize=VALUE_LABEL_FONTSIZE,
                weight="bold",
                color=INK,
            )

        ax.axhline(0, color=INK_2, linewidth=1.0)
        ax.set_xticks([3, 4, 5])
        ax.set_xlim(2.7, 5.3)
        ax.set_xlabel("SID depth (digits)")
        ax.set_ylabel(f"{metric} gap: diffusion-family − autoregressive")
        ax.set_title(metric)
        despine(ax)

    tokenizer_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=TOKENIZER_COLORS[t], label=t)
        for t in TOKENIZER_ORDER
    ]
    codebook_handles = [
        Line2D(
            [0],
            [0],
            marker=CODEBOOK_MARKERS[c],
            linestyle="",
            markerfacecolor="white" if c == 256 else "#777777",
            markeredgecolor="#555555",
            label=f"codebook {c}" + (" †" if c == 256 else ""),
        )
        for c in (128, 256, 512)
    ]
    mean_handle = Line2D([0], [0], color=INK, marker="D", label="depth mean")
    fig.legend(
        handles=tokenizer_handles + codebook_handles + [mean_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
    )
    add_panel_labels(axes)
    add_header(
        fig,
        "Paradigm performance gap by SID depth",
        "Pairing rows at the same nominal tokenizer × codebook × depth setting controls SID design within every comparison.",
    )
    add_footer(
        fig,
        "Positive values favor the diffusion-family row. The workbook calls it ‘Mask Diffusion’ at codebooks 128/512 and ‘DiffGRM’ at 256; equivalence is assumed only for this paradigm-level view. No seed variance is available.",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.72, bottom=0.17, wspace=0.25)
    save_figure(fig, output_dir, "fig06_paradigm_gap_by_depth")


def _display_model(model: str) -> str:
    return {
        "Qwen2.5-1.5B-grpo": "Qwen2.5-1.5B + GRPO",
        "Mask Diffusion": "Mask Diffusion",
        "DiffGRM": "DiffGRM",
    }.get(model, model)


def fig07_model_training_ablation(ablation: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.3), sharex=True)

    for ax, tokenizer in zip(axes, TOKENIZER_ORDER):
        frame = ablation[ablation["tokenizer"] == tokenizer].sort_values(
            "HR@10", ascending=False
        )
        y_positions = np.arange(len(frame))
        for y, row in zip(y_positions, frame.itertuples(index=False)):
            # Metric names contain punctuation and are sanitized by itertuples;
            # indexing the frame below keeps this explicit and version-stable.
            values = frame.loc[frame["model"] == row.model, ["HR@10", "NDCG@10"]].iloc[0]
            ax.plot(
                [values["NDCG@10"], values["HR@10"]],
                [y, y],
                color="#B8B7B1",
                linewidth=2.0,
                zorder=1,
            )
            ax.scatter(values["HR@10"], y, s=62, color="#4477AA", zorder=3)
            ax.scatter(
                values["NDCG@10"], y, s=56, marker="D", color="#AA3377", zorder=3
            )
            ax.text(
                values["HR@10"] + 0.28,
                y,
                f"{values['HR@10']:.2f}",
                va="center",
                fontsize=VALUE_LABEL_FONTSIZE,
                color="#315783",
            )
        ax.set_yticks(y_positions)
        ax.set_yticklabels([_display_model(model) for model in frame["model"]])
        ax.invert_yaxis()
        collision = float(frame["collision_pct"].iloc[0])
        ax.set_title(
            f"{tokenizer}\nnominal 3d × cb256 · {collision:.2f}% collision †",
            color=TOKENIZER_COLORS[tokenizer],
        )
        ax.set_xlabel("Reported score")
        ax.set_xlim(4, 25.5)
        despine(ax)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#4477AA", label="HR@10"),
        Line2D(
            [0], [0], marker="D", linestyle="", color="#AA3377", label="NDCG@10"
        ),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2)
    add_panel_labels(axes, x=0.015, y=0.96)
    add_header(
        fig,
        "Model and training ablations at 3 digits × codebook 256",
        "Rows mix backbone/scale, GRPO tuning, and generator paradigm; all use the nominal codebook-256 × 3-digit configuration.",
    )
    add_footer(
        fig,
        "† Collision-table values are flagged old artifacts and may not describe the exact SIDs used for these metric rows. Within-panel runs share a nominal SID design, not one isolated intervention; no seed variance is available.",
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.79, bottom=0.15, wspace=0.48)
    save_figure(fig, output_dir, "fig07_model_training_ablation")


def fig08_full_configuration_grid(
    means: pd.DataFrame, gaps: pd.DataFrame, cross: pd.DataFrame, output_dir: Path
) -> None:
    grid = means.merge(
        cross[["tokenizer", "depth", "codebook", "Office"]],
        on=["tokenizer", "depth", "codebook"],
        validate="one_to_one",
    ).merge(
        gaps[["tokenizer", "depth", "codebook", "HR@10_gap"]],
        on=["tokenizer", "depth", "codebook"],
        validate="one_to_one",
    )

    # Recover paradigm-specific HR@10 from the mean and paired gap.
    grid["AR HR@10"] = grid["HR@10"] - grid["HR@10_gap"] / 2
    grid["Diffusion HR@10"] = grid["HR@10"] + grid["HR@10_gap"] / 2
    grid = grid.sort_values(["tokenizer", "codebook", "depth"]).reset_index(drop=True)
    row_labels = [
        f"{row.tokenizer}  ·  cb{row.codebook}  ·  {row.depth}d"
        for row in grid.itertuples(index=False)
    ]

    columns = [
        ("collision_pct", "Industrial\ncollision", MATRIX_SEQUENTIAL, LogNorm(0.3, 60), ".2f", "%"),
        ("Office", "Office\ncollision", MATRIX_SEQUENTIAL, LogNorm(0.3, 60), ".2f", "%"),
        ("AR HR@10", "AR\nHR@10", MATRIX_SEQUENTIAL, Normalize(7, 25), ".2f", ""),
        ("Diffusion HR@10", "Diffusion\nHR@10", MATRIX_SEQUENTIAL, Normalize(7, 25), ".2f", ""),
        (
            "HR@10_gap",
            "Paired gap\nDiff − AR",
            MATRIX_DIVERGING,
            TwoSlopeNorm(vmin=-8, vcenter=0, vmax=2),
            "+.2f",
            "",
        ),
    ]
    fig, axes = plt.subplots(
        1,
        len(columns),
        figsize=(11.6, 12.6),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    for index, (ax, (column, title, cmap, norm, number_format, suffix)) in enumerate(
        zip(axes, columns)
    ):
        values = grid[column].to_numpy(dtype=float)[:, None]
        ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
        ax.grid(False, which="both")
        ax.set_xticks([0])
        ax.set_xticklabels([title])
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=9)
        ax.set_yticks(np.arange(len(grid)))
        if index == 0:
            ax.set_yticklabels(row_labels, fontsize=10)
        ax.tick_params(axis="y", length=0)
        for row_index, value in enumerate(values[:, 0]):
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.53 else INK
            ax.text(
                0,
                row_index,
                f"{value:{number_format}}{suffix}",
                ha="center",
                va="center",
                fontsize=9.2,
                color=text_color,
                weight="bold" if column == "HR@10_gap" else "normal",
            )
        for separator in (8.5, 17.5):
            ax.axhline(separator, color=SURFACE, linewidth=4)
        for spine in ax.spines.values():
            spine.set_visible(False)

    add_header(
        fig,
        "Configuration-level result matrix",
        "Collision rates, paradigm-specific HR@10, and the paired gap are aligned row-by-row; each column uses its own color scale.",
    )
    add_footer(
        fig,
        "Rows are sorted by tokenizer, codebook, then depth. Industrial performance is inferred from workbook placement. Values, not cross-column color intensity, should be compared across measures.",
    )
    fig.subplots_adjust(left=0.34, right=0.985, top=0.985, bottom=0.065)
    save_figure(fig, output_dir, "fig08_full_configuration_grid")


def fig09_factorial_main_effects(
    effects: pd.DataFrame, output_dir: Path
) -> None:
    """Show the balanced marginal effects that the pooled plots obscure."""

    factors = [
        ("tokenizer", TOKENIZER_ORDER, "Tokenizer"),
        ("depth", [3, 4, 5], "SID depth (digits)"),
        ("codebook", [128, 256, 512], "Codebook size"),
    ]
    metrics = ["HR@10", "NDCG@10"]
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.7), sharey="row")
    offsets = {"Autoregressive": -0.10, "Diffusion-family": 0.10}

    for row_index, metric in enumerate(metrics):
        for col_index, (factor, levels, title) in enumerate(factors):
            ax = axes[row_index, col_index]
            panel = effects[
                (effects["metric"] == metric) & (effects["factor"] == factor)
            ]
            x = np.arange(len(levels), dtype=float)
            for family in ("Autoregressive", "Diffusion-family"):
                frame = panel[panel["family"] == family].sort_values("level_index")
                means = frame["mean"].to_numpy(dtype=float)
                lower = means - frame["minimum"].to_numpy(dtype=float)
                upper = frame["maximum"].to_numpy(dtype=float) - means
                positions = x + offsets[family]
                ax.errorbar(
                    positions,
                    means,
                    yerr=np.vstack([lower, upper]),
                    fmt=FAMILY_MARKERS[family],
                    color=FAMILY_COLORS[family],
                    markersize=7.5,
                    linewidth=1.8,
                    elinewidth=1.2,
                    capsize=4,
                    capthick=1.1,
                    zorder=3,
                )
                ax.plot(
                    positions,
                    means,
                    color=FAMILY_COLORS[family],
                    linewidth=1.2,
                    alpha=0.65,
                )
            ax.set_xticks(x)
            ax.set_xticklabels([str(level) for level in levels])
            ax.set_xlabel(title)
            if row_index == 0:
                ax.set_title(title)
            if col_index == 0:
                ax.set_ylabel(f"Reported {metric}")
            despine(ax)

    handles = [
        Line2D(
            [0],
            [0],
            color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family],
            label=family,
        )
        for family in ("Autoregressive", "Diffusion-family")
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
    )
    add_panel_labels(axes)
    add_header(
        fig,
        "Balanced main effects of SID design choices",
        "Means use the complete 3 × 3 × 3 factorial grid separately for each generator paradigm.",
    )
    add_footer(
        fig,
        "Whiskers show the observed minimum-to-maximum range across the other two design factors (n = 9 cells per point); they are descriptive ranges, not confidence intervals or seed uncertainty.",
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, top=0.86, bottom=0.13, wspace=0.18, hspace=0.32
    )
    save_figure(fig, output_dir, "fig09_factorial_main_effects")


def fig10_cross_dataset_paradigm_gap(
    gaps: pd.DataFrame, output_dir: Path
) -> None:
    """Expose the dataset-by-paradigm interaction at the shared cb256 grid."""

    dataset_colors = {"Industrial": "#3B6FB6", "Office": "#EE7733"}
    dataset_offsets = {"Industrial": -0.08, "Office": 0.08}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8))

    for ax, metric in zip(axes, ("HR@10", "NDCG@10")):
        gap_column = f"{metric}_gap"
        for dataset in ("Industrial", "Office"):
            frame = gaps[gaps["dataset"] == dataset]
            offset = dataset_offsets[dataset]
            color = dataset_colors[dataset]
            for tokenizer in TOKENIZER_ORDER:
                path = frame[frame["tokenizer"] == tokenizer].sort_values("depth")
                x = path["depth"].to_numpy(dtype=float) + offset
                ax.plot(
                    x,
                    path[gap_column],
                    color=color,
                    linewidth=1.0,
                    alpha=0.28,
                    zorder=1,
                )
                ax.scatter(
                    x,
                    path[gap_column],
                    marker=TOKENIZER_MARKERS[tokenizer],
                    s=72,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=3,
                )

            means = frame.groupby("depth", observed=True)[gap_column].mean()
            mean_x = means.index.to_numpy(dtype=float) + offset
            ax.plot(
                mean_x,
                means.to_numpy(dtype=float),
                color=color,
                marker="D",
                markersize=7,
                linewidth=2.6,
                zorder=4,
            )
            for x_value, value in zip(mean_x, means.to_numpy(dtype=float)):
                ax.annotate(
                    f"{value:+.2f}",
                    (x_value, value),
                    xytext=(0, 9 if value >= 0 else -15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=VALUE_LABEL_FONTSIZE,
                    weight="bold",
                    color=color,
                )

        ax.axhline(0, color=INK_2, linewidth=1.0)
        ax.set_xticks([3, 4, 5])
        ax.set_xlim(2.65, 5.35)
        ax.set_xlabel("SID depth (digits)")
        ax.set_ylabel(f"{metric} gap: diffusion-family − autoregressive")
        ax.set_title(metric)
        despine(ax)

    dataset_handles = [
        Line2D(
            [0],
            [0],
            color=dataset_colors[dataset],
            marker="D",
            linewidth=2.4,
            label=f"{dataset} mean",
        )
        for dataset in ("Industrial", "Office")
    ]
    tokenizer_handles = [
        Line2D(
            [0],
            [0],
            color="#666666",
            marker=TOKENIZER_MARKERS[tokenizer],
            linestyle="",
            label=tokenizer,
        )
        for tokenizer in TOKENIZER_ORDER
    ]
    fig.legend(
        handles=dataset_handles + tokenizer_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )
    add_panel_labels(axes)
    add_header(
        fig,
        "Cross-dataset paradigm interaction at codebook 256",
        "The complete Office table reverses the depth-dependent generator gap seen in the Industrial results.",
    )
    add_footer(
        fig,
        "Positive values favor DiffGRM/diffusion-family. Points are tokenizer-specific cells; diamonds are means across the three tokenizers. Office source: research-notes PDF, table dated 2026-07-21. No seed variance is available.",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.74, bottom=0.17, wspace=0.25)
    save_figure(fig, output_dir, "fig10_cross_dataset_paradigm_gap")


def prepare_resource_table(resources: pd.DataFrame) -> pd.DataFrame:
    """Validate and add clearly named derived resource quantities."""

    required = {
        "variant_id",
        "method",
        "codebooks",
        "codebook_size",
        "ndcg_at_10",
        "train_wall_seconds",
        "eval_seconds",
        "evaluation_examples",
        "evaluation_num_beams",
        "evaluation_seconds_per_example",
        "evaluation_allocated_gpu_hours",
        "inference_flops_status",
        "hf_total_flops",
        "gpu_count",
        "gpu_model",
        "mean_gpu_util_percent",
        "peak_gpu_memory_allocated_gib",
        "measured_gpu_energy_kwh",
        "hf_estimated_achieved_tflops_per_gpu",
    }
    missing = required - set(resources.columns)
    if missing:
        raise ValueError(f"Resource table is missing required columns: {sorted(missing)}")

    frame = resources.copy()
    numeric = required - {
        "variant_id",
        "method",
        "gpu_model",
        "inference_flops_status",
    }
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    key = ["method", "codebooks", "codebook_size"]
    expected = {
        (method, depth, codebook_size)
        for method in ("rqvae", "rqkmeans", "MQ")
        for depth in (3, 4, 5)
        for codebook_size in (128, 512)
    }
    actual = set(frame[key].itertuples(index=False, name=None))
    if len(frame) != 18 or frame.duplicated(key).any() or actual != expected:
        raise ValueError("Resource table must contain the complete 18-run current grid")
    if set(frame["gpu_model"]) != {"NVIDIA A100-SXM4-40GB"}:
        raise ValueError("Resource table contains an unexpected accelerator model")
    if set(frame["gpu_count"].astype(int)) != {4}:
        raise ValueError("Resource table contains an unexpected GPU count")
    if set(frame["evaluation_examples"].astype(int)) != {3681}:
        raise ValueError("Resource table contains an unexpected evaluation-set size")
    if set(frame["evaluation_num_beams"].astype(int)) != {50}:
        raise ValueError("Resource table contains an unexpected evaluation beam count")
    if not frame["inference_flops_status"].str.startswith("not measured").all():
        raise ValueError("Inference-FLOP provenance is missing or ambiguous")

    frame["hf_training_eflop"] = frame["hf_total_flops"] / 1e18
    frame["wall_gpu_hours"] = (
        frame["train_wall_seconds"] * frame["gpu_count"] / 3600
    )
    frame["evaluation_ms_per_example"] = (
        frame["evaluation_seconds_per_example"] * 1000
    )
    frame["total_allocated_gpu_hours"] = (
        frame["wall_gpu_hours"] + frame["evaluation_allocated_gpu_hours"]
    )
    return frame.sort_values(key).reset_index(drop=True)


def resource_summary(resources: pd.DataFrame) -> pd.DataFrame:
    measures = [
        ("hf_training_eflop", "HF-estimated training FLOPs", "EFLOP"),
        ("wall_gpu_hours", "Training wall GPU-hours", "GPU-hours"),
        ("eval_seconds", "Measured end-to-end evaluation wall time", "s"),
        (
            "evaluation_ms_per_example",
            "End-to-end evaluation wall time per example",
            "ms/example",
        ),
        (
            "evaluation_allocated_gpu_hours",
            "Allocated evaluation GPU-hours",
            "GPU-hours",
        ),
        (
            "total_allocated_gpu_hours",
            "Training plus evaluation allocated GPU-hours",
            "GPU-hours",
        ),
        ("measured_gpu_energy_kwh", "Sampled training GPU energy", "kWh"),
        ("peak_gpu_memory_allocated_gib", "Peak allocated memory", "GiB/GPU"),
        ("mean_gpu_util_percent", "Mean GPU utilization", "%"),
        (
            "hf_estimated_achieved_tflops_per_gpu",
            "HF-estimated achieved throughput",
            "TFLOP/s/GPU",
        ),
    ]
    additive = {
        "hf_training_eflop",
        "wall_gpu_hours",
        "evaluation_allocated_gpu_hours",
        "total_allocated_gpu_hours",
        "measured_gpu_energy_kwh",
    }
    records: list[dict[str, object]] = []
    for column, measure, unit in measures:
        values = resources[column].astype(float)
        records.append(
            {
                "measure": measure,
                "source_column": column,
                "unit": unit,
                "n_runs": len(values),
                "minimum": values.min(),
                "median": values.median(),
                "mean": values.mean(),
                "maximum": values.max(),
                "total": values.sum() if column in additive else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def _pareto_mask(frame: pd.DataFrame, cost: str, score: str) -> pd.Series:
    mask = []
    for row in frame.itertuples(index=False):
        row_cost = float(getattr(row, cost))
        row_score = float(getattr(row, score))
        dominated = (
            (frame[cost] <= row_cost)
            & (frame[score] >= row_score)
            & ((frame[cost] < row_cost) | (frame[score] > row_score))
        ).any()
        mask.append(not dominated)
    return pd.Series(mask, index=frame.index)


def fig11_training_compute_resources(
    resources: pd.DataFrame, output_dir: Path
) -> None:
    """Summarize measured accelerator use and the accuracy-resource frontier."""

    method_display = {
        "rqvae": "RQ-VAE",
        "rqkmeans": "RQ-KMeans",
        "MQ": "MQ",
    }
    method_offsets = {"rqvae": -0.13, "rqkmeans": 0.0, "MQ": 0.13}
    size_offsets = {128: -0.025, 512: 0.025}
    panels = [
        ("hf_training_eflop", "HF-estimated training FLOPs (EFLOP)"),
        ("eval_seconds", "Measured end-to-end evaluation time (s)"),
        ("measured_gpu_energy_kwh", "Sampled training GPU energy (kWh)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.9))

    for ax, (column, ylabel) in zip(axes.flat[:3], panels):
        for method in ("rqvae", "rqkmeans", "MQ"):
            for codebook_size in (128, 512):
                path = resources[
                    (resources["method"] == method)
                    & (resources["codebook_size"] == codebook_size)
                ].sort_values("codebooks")
                x = (
                    path["codebooks"].to_numpy(dtype=float)
                    + method_offsets[method]
                    + size_offsets[codebook_size]
                )
                color = TOKENIZER_COLORS[method_display[method]]
                ax.plot(x, path[column], color=color, linewidth=1.0, alpha=0.25)
                ax.scatter(
                    x,
                    path[column],
                    s=74,
                    marker=CODEBOOK_MARKERS[codebook_size],
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.92,
                    zorder=3,
                )
        ax.set_xticks([3, 4, 5])
        ax.set_xlim(2.68, 5.32)
        ax.set_xlabel("SID depth (digits)")
        ax.set_ylabel(ylabel)
        despine(ax)

    pareto_ax = axes[1, 1]
    for method in ("rqvae", "rqkmeans", "MQ"):
        color = TOKENIZER_COLORS[method_display[method]]
        for codebook_size in (128, 512):
            frame = resources[
                (resources["method"] == method)
                & (resources["codebook_size"] == codebook_size)
            ]
            pareto_ax.scatter(
                frame["total_allocated_gpu_hours"],
                frame["ndcg_at_10"],
                s=78,
                marker=CODEBOOK_MARKERS[codebook_size],
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.92,
                zorder=3,
            )

    pareto = resources[
        _pareto_mask(resources, "total_allocated_gpu_hours", "ndcg_at_10")
    ].sort_values("total_allocated_gpu_hours")
    pareto_ax.plot(
        pareto["total_allocated_gpu_hours"],
        pareto["ndcg_at_10"],
        color=INK,
        linestyle="--",
        linewidth=1.4,
        zorder=2,
    )
    for index, row in enumerate(pareto.itertuples(index=False)):
        label = f"{method_display[row.method]} · {int(row.codebooks)}d × {int(row.codebook_size)}"
        pareto_ax.annotate(
            label,
            (row.total_allocated_gpu_hours, row.ndcg_at_10),
            xytext=(6, 8 if index % 2 == 0 else -14),
            textcoords="offset points",
            fontsize=VALUE_LABEL_FONTSIZE,
            color=TOKENIZER_COLORS[method_display[row.method]],
        )
    pareto_ax.set_xlabel("Training + evaluation allocated GPU-hours (4 × A100)")
    pareto_ax.set_ylabel("NDCG@10")
    pareto_ax.set_title("Accuracy–resource Pareto frontier")
    despine(pareto_ax)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=TOKENIZER_COLORS[method_display[method]],
            label=method_display[method],
        )
        for method in ("rqvae", "rqkmeans", "MQ")
    ]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker=CODEBOOK_MARKERS[size],
            linestyle="",
            color="#666666",
            label=f"codebook {size}",
        )
        for size in (128, 512)
    ]
    pareto_handle = Line2D(
        [0], [0], color=INK, linestyle="--", label="non-dominated frontier"
    )
    fig.legend(
        handles=method_handles + size_handles + [pareto_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )
    add_panel_labels(axes)
    add_header(
        fig,
        "Training compute, accelerator use, and resource efficiency",
        "Eighteen completed Industrial next-item SFT runs use the same full-tuning recipe on four A100-SXM4-40GB GPUs.",
    )
    add_footer(
        fig,
        "(a) FLOPs are the Hugging Face training estimate, not hardware counters. (b) Measured wall time covers 3,681 examples with 50-beam constrained decoding on 4 GPUs and includes loading, split/merge, and scoring—not generation alone.\nExisting logs contain no inference FLOP counter or inference-only telemetry; those require a profiled rerun. Energy is sampled training-GPU energy only. No codebook-256 resource trace is available.",
    )
    fig.subplots_adjust(
        left=0.085, right=0.985, top=0.79, bottom=0.16, wspace=0.24, hspace=0.34
    )
    save_figure(fig, output_dir, "fig11_training_compute_resources")


def fig12_sid_depth_tradeoff(
    collisions: pd.DataFrame,
    gaps: pd.DataFrame,
    resources: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Join addressability, paired model behavior, and measured cost by depth.

    The common codebook-128/512 grid is used deliberately: those collision
    values are current runs and those are the configurations with resource
    traces.  Codebook 256 is therefore neither mixed into the comparison nor
    silently imputed.
    """

    method_display = {"rqvae": "RQ-VAE", "rqkmeans": "RQ-KMeans", "MQ": "MQ"}
    method_lookup = {display: method for method, display in method_display.items()}
    method_offsets = {"RQ-VAE": -0.13, "RQ-KMeans": 0.0, "MQ": 0.13}
    codebook_offsets = {128: -0.025, 512: 0.025}

    current_collisions = collisions[
        (collisions["dataset"] == "Industrial")
        & (collisions["codebook"].isin([128, 512]))
    ].copy()
    current_gaps = gaps[gaps["codebook"].isin([128, 512])].copy()
    if len(current_collisions) != 18 or len(current_gaps) != 18 or len(resources) != 18:
        raise ValueError("SID-depth trade-off requires three complete 18-cell grids")

    expected = {
        (tokenizer, depth, codebook)
        for tokenizer in TOKENIZER_ORDER
        for depth in (3, 4, 5)
        for codebook in (128, 512)
    }
    collision_keys = set(
        current_collisions[["tokenizer", "depth", "codebook"]].itertuples(
            index=False, name=None
        )
    )
    gap_keys = set(
        current_gaps[["tokenizer", "depth", "codebook"]].itertuples(
            index=False, name=None
        )
    )
    resource_keys = {
        (method_display[str(method)], int(depth), int(codebook))
        for method, depth, codebook in resources[
            ["method", "codebooks", "codebook_size"]
        ].itertuples(index=False, name=None)
    }
    if collision_keys != expected or gap_keys != expected or resource_keys != expected:
        raise ValueError("SID-depth trade-off grids are not configuration-aligned")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.1))
    columns = ["collision_pct", "HR@10_gap", "evaluation_ms_per_example"]

    for tokenizer in TOKENIZER_ORDER:
        for codebook in (128, 512):
            offset = method_offsets[tokenizer] + codebook_offsets[codebook]
            color = TOKENIZER_COLORS[tokenizer]
            marker = CODEBOOK_MARKERS[codebook]

            collision_path = current_collisions[
                (current_collisions["tokenizer"] == tokenizer)
                & (current_collisions["codebook"] == codebook)
            ].sort_values("depth")
            gap_path = current_gaps[
                (current_gaps["tokenizer"] == tokenizer)
                & (current_gaps["codebook"] == codebook)
            ].sort_values("depth")
            resource_path = resources[
                (resources["method"] == method_lookup[tokenizer])
                & (resources["codebook_size"] == codebook)
            ].sort_values("codebooks")

            for ax, frame, x_column, y_column in zip(
                axes,
                (collision_path, gap_path, resource_path),
                ("depth", "depth", "codebooks"),
                columns,
            ):
                x = frame[x_column].to_numpy(dtype=float) + offset
                y = frame[y_column].to_numpy(dtype=float)
                ax.plot(x, y, color=color, linewidth=1.15, alpha=0.28, zorder=1)
                ax.scatter(
                    x,
                    y,
                    s=72,
                    marker=marker,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.94,
                    zorder=3,
                )

    gap_means = current_gaps.groupby("depth", observed=True)["HR@10_gap"].mean()
    evaluation_means = resources.groupby("codebooks", observed=True)[
        "evaluation_ms_per_example"
    ].mean()
    for ax, means, value_format, vertical_offset in (
        (axes[1], gap_means, "{:+.2f}", -17),
        (axes[2], evaluation_means, "{:.1f}", 10),
    ):
        x = means.index.to_numpy(dtype=float)
        y = means.to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=INK,
            marker="D",
            markersize=7,
            linewidth=2.5,
            zorder=5,
        )
        for x_value, y_value in zip(x, y):
            ax.annotate(
                value_format.format(y_value),
                (x_value, y_value),
                xytext=(0, vertical_offset),
                textcoords="offset points",
                ha="center",
                fontsize=VALUE_LABEL_FONTSIZE,
                weight="bold",
                color=INK,
            )

    axes[0].set_yscale("log")
    axes[0].set_ylim(0.28, 60)
    axes[0].set_yticks([0.3, 1, 3, 10, 30, 60])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    axes[0].set_ylabel("Collision rate (log scale)")
    axes[0].set_title("Collision rate")

    axes[1].axhline(0, color=INK_2, linewidth=1.0)
    axes[1].set_ylabel("HR@10 gap\nMask diffusion − autoregressive")
    axes[1].set_title("Paired HR@10 gap")

    axes[2].set_ylabel("End-to-end evaluation wall time\n(ms/example)")
    axes[2].set_title("Evaluation time")

    for ax in axes:
        ax.set_xticks([3, 4, 5])
        ax.set_xlim(2.68, 5.32)
        ax.set_xlabel("SID depth (digits)")
        despine(ax)

    tokenizer_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=TOKENIZER_COLORS[tokenizer],
            label=tokenizer,
        )
        for tokenizer in TOKENIZER_ORDER
    ]
    codebook_handles = [
        Line2D(
            [0],
            [0],
            marker=CODEBOOK_MARKERS[codebook],
            linestyle="",
            color="#666666",
            label=f"codebook {codebook}",
        )
        for codebook in (128, 512)
    ]
    mean_handle = Line2D(
        [0], [0], color=INK, marker="D", linewidth=2.2, label="depth mean"
    )
    fig.legend(
        handles=tokenizer_handles + codebook_handles + [mean_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )
    add_panel_labels(axes)
    add_header(
        fig,
        "SID-depth trade-off on the current Industrial grid",
        "Deeper SIDs reduce collisions, widen the paired Mask-Diffusion deficit, and increase measured evaluation time.",
    )
    add_footer(
        fig,
        "Current Industrial codebooks 128/512 only. Thin paths are tokenizer × codebook configurations; diamonds are descriptive depth means. Evaluation covers 3,681 examples, 50-beam constrained decoding, and 4 × A100, including loading, split/merge, and scoring—not generation alone. No seed variance is available.",
    )
    fig.subplots_adjust(left=0.075, right=0.988, top=0.75, bottom=0.20, wspace=0.31)
    save_figure(fig, output_dir, "fig12_sid_depth_tradeoff")


def write_report(
    tables: dict[str, pd.DataFrame],
    derived: dict[str, pd.DataFrame],
    output_dir: Path,
    workbook: Path,
    research_notes: Path | None = None,
    resources: pd.DataFrame | None = None,
) -> None:
    collisions = tables["collisions"]
    metrics = tables["metrics"]
    means = derived["configuration_means"]
    gaps = derived["paradigm_gaps"]
    cross = derived["cross_dataset_collisions"]
    within = derived["within_tokenizer_associations"]

    hr_rho, hr_p = _spearman(means["collision_pct"], means["HR@10"])
    ndcg_rho, ndcg_p = _spearman(means["collision_pct"], means["NDCG@10"])
    sensitivity = means[means["codebook"] != 256]
    hr_sensitivity, hr_sensitivity_p = _spearman(
        sensitivity["collision_pct"], sensitivity["HR@10"]
    )
    cross_rho, cross_p = _spearman(cross["Industrial"], cross["Office"])

    metric_rhos = [
        _spearman(means["collision_pct"], means[metric])[0] for metric in METRICS
    ]
    metric_corr = means[METRICS].corr(method="spearman")
    upper = metric_corr.where(np.triu(np.ones(metric_corr.shape), k=1).astype(bool)).stack()

    within_hr = within[within["metric"] == "HR@10"].set_index("tokenizer")
    depth_gaps = gaps.groupby("depth")["HR@10_gap"].mean()
    five_digit = gaps[gaps["depth"] == 5]["HR@10_gap"]
    five_wilcoxon = stats.wilcoxon(five_digit, method="exact")

    outlier = cross[
        (cross["tokenizer"] == "RQ-KMeans")
        & (cross["depth"] == 5)
        & (cross["codebook"] == 512)
    ].iloc[0]
    digest = hashlib.sha256(workbook.read_bytes()).hexdigest()

    office_scope = ""
    cross_dataset_finding = ""
    cross_dataset_limit = ""
    if "cross_dataset_paradigm_gaps_codebook256" in derived:
        cross_gaps = derived["cross_dataset_paradigm_gaps_codebook256"]
        agreement = derived["cross_dataset_performance_agreement_codebook256"]
        gap_means = cross_gaps.groupby(["dataset", "depth"])["HR@10_gap"].mean()
        transfer_rho, transfer_p = _spearman(
            agreement["HR@10_Industrial"], agreement["HR@10_Office"]
        )
        notes_digest = (
            hashlib.sha256(research_notes.read_bytes()).hexdigest()
            if research_notes is not None
            else "not recorded"
        )
        office_scope = (
            "- Office performance comparison: **18 rows** = 3 tokenizers × 3 depths × "
            "2 paradigms at codebook 256, extracted from research-notes PDF pages 1–3 "
            f"(SHA-256 `{notes_digest}`).\n"
        )
        cross_dataset_finding = (
            "7. **The Industrial paradigm-depth result does not transfer to Office at "
            "codebook 256.** Mean diffusion-family minus autoregressive HR@10 is "
            f"**{gap_means.loc[('Industrial', 3)]:+.2f}/{gap_means.loc[('Industrial', 4)]:+.2f}/"
            f"{gap_means.loc[('Industrial', 5)]:+.2f}** on Industrial at depths 3/4/5, "
            f"but **{gap_means.loc[('Office', 3)]:+.2f}/{gap_means.loc[('Office', 4)]:+.2f}/"
            f"{gap_means.loc[('Office', 5)]:+.2f}** on Office. Row-level HR@10 rank "
            f"agreement is ρ = **{transfer_rho:+.3f}** ({_p_text(transfer_p)}; n = 18)."
        )
        cross_dataset_limit = (
            "- The Office codebook-256 heading is explicit for RQ-VAE. Its continuation "
            "through the RQ-KMeans and Parallel-SID subtables is a section-layout inference, "
            "recorded row-by-row in `office_core_codebook256.csv`.\n"
        )

    resource_scope = ""
    resource_finding = ""
    resource_limit = ""
    if resources is not None and not resources.empty:
        pareto = resources[
            _pareto_mask(resources, "total_allocated_gpu_hours", "ndcg_at_10")
        ]
        resource_scope = (
            "- Training-resource comparison: **18 completed Industrial next-item "
            "SFT runs** = 3 SID tokenizers × 3 depths × 2 codebook sizes (128 and "
            "512), all on 4 × NVIDIA A100-SXM4-40GB GPUs. Each retained evaluation "
            "contains 3,681 examples and uses 50-beam constrained decoding.\n"
        )
        resource_finding = (
            "8. **The available SFT runs occupy a fairly narrow compute envelope, "
            "but wall time still varies.** Hugging Face estimated training compute spans "
            f"**{resources['hf_training_eflop'].min():.3f}–"
            f"{resources['hf_training_eflop'].max():.3f} EFLOP**, while sweep wall time "
            f"spans **{resources['wall_gpu_hours'].min():.2f}–"
            f"{resources['wall_gpu_hours'].max():.2f} GPU-hours** and sampled GPU energy "
            f"spans **{resources['measured_gpu_energy_kwh'].min():.2f}–"
            f"{resources['measured_gpu_energy_kwh'].max():.2f} kWh**. Measured end-to-end "
            f"evaluation time is **{resources['eval_seconds'].min():.0f}–"
            f"{resources['eval_seconds'].max():.0f} s** "
            f"(**{resources['evaluation_ms_per_example'].min():.1f}–"
            f"{resources['evaluation_ms_per_example'].max():.1f} ms/example**). "
            f"**{len(pareto)} of 18** configurations are non-dominated on the observed "
            "NDCG@10 versus total training-plus-evaluation allocation trade-off."
        )
        resource_limit = (
            "- Resource data cover autoregressive SFT only and omit codebook 256; they "
            "must not be used as a direct AR-versus-DiffGRM efficiency comparison. "
            "`total_flos` is the Hugging Face analytical estimate (6 × padded tokens × "
            "non-embedding parameters), not a hardware-counter measurement. Energy "
            "integrates sampled training-GPU power and excludes CPUs, cooling, evaluation, "
            "and PUE. Existing evaluation artifacts record end-to-end wall time but no "
            "inference FLOP counter or inference-only telemetry, so inference FLOPs cannot "
            "be reconstructed without a profiled rerun.\n"
        )

    figure_rows = """| [`fig01_collision_design_space`](fig01_collision_design_space.png) | How do tokenizer, depth, codebook, and dataset relate to collision? |
| [`fig02_cross_dataset_collision`](fig02_cross_dataset_collision.png) | Does a configuration's collision behavior transfer across datasets? |
| [`fig03_collision_vs_reported_metrics`](fig03_collision_vs_reported_metrics.png) | How strongly do collision and headline metrics co-vary? |
| [`fig04_within_tokenizer_association`](fig04_within_tokenizer_association.png) | Does the relationship persist within each tokenizer? |
| [`fig05_metric_robustness`](fig05_metric_robustness.png) | Is the association consistent across metrics, paradigms, and artifact sensitivity? |
| [`fig06_paradigm_gap_by_depth`](fig06_paradigm_gap_by_depth.png) | How does the paired generator-paradigm gap change with SID depth? |
| [`fig07_model_training_ablation`](fig07_model_training_ablation.png) | What model/training comparisons are available at the shared nominal setting? |
| [`fig08_full_configuration_grid`](fig08_full_configuration_grid.png) | What are the aligned values for every core SID configuration? |
| [`fig09_factorial_main_effects`](fig09_factorial_main_effects.png) | What are the balanced marginal effects of tokenizer, depth, and codebook? |"""
    if "cross_dataset_paradigm_gaps_codebook256" in derived:
        figure_rows += (
            "\n| [`fig10_cross_dataset_paradigm_gap`](fig10_cross_dataset_paradigm_gap.png) "
            "| Does the generator-paradigm gap transfer from Industrial to Office? |"
        )
    if resources is not None and not resources.empty:
        figure_rows += (
            "\n| [`fig11_training_compute_resources`](fig11_training_compute_resources.png) "
            "| How do estimated training compute, measured evaluation time, and accuracy trade off? |"
        )
        figure_rows += (
            "\n| [`fig12_sid_depth_tradeoff`](fig12_sid_depth_tradeoff.png) "
            "| How does SID depth jointly affect collision, the paired model gap, and measured evaluation cost? |"
        )

    report = f"""# Collision-rate and reported-performance analysis

Source workbook: `{workbook.name}`  
SHA-256: `{digest}`

Regenerate from the repository root:

```bash
.conda/bin/python scripts/diagrams/new/make_all.py
```

## Plotting implementation

- All plots are generated in Python with **Matplotlib {matplotlib.__version__}** using its non-interactive `Agg` backend; no browser charting or image-generation tool is involved.
- Pandas and NumPy prepare the tables, SciPy computes the reported statistics, OpenPyXL reads the workbook, pypdf extracts the Office table, and the W&B protobuf reader recovers training telemetry.
- Figure-level titles are intentionally omitted from the image files; panel titles, axes, legends, and the paper/report caption carry the context.

## Scope and data treatment

- Collision grid: **{len(collisions)} rows** = 2 datasets × 3 tokenizers × 3 depths × 3 codebooks.
- Populated performance results: **{len(metrics)} rows**. The balanced analysis uses **54 core rows** = 27 SID configurations × 2 generator paradigms; **{len(metrics) - 54} additional model/training-ablation rows** are isolated in figure 7.
{office_scope}{resource_scope}- Every collision/performance association uses the **27 SID configurations** as the statistical unit by averaging the paired paradigms. This avoids counting the same collision value twice.
- The performance section does not explicitly name its dataset. It is treated as **Industrial by workbook placement**, which is an inference.
- Collision is `(N − unique full SIDs) / N`: the count is excess SID assignments, not the number of all items that belong to collided groups.

## Main findings

1. **Collision rate and reported ranking metrics move together.** Across the 27 paired-configuration means, collision versus HR@10 has Spearman ρ = **{hr_rho:.3f}** ({_p_text(hr_p)}), and collision versus NDCG@10 has ρ = **{ndcg_rho:.3f}** ({_p_text(ndcg_p)}). Across all six reported metrics, ρ spans **{min(metric_rhos):.3f}–{max(metric_rhos):.3f}**.

2. **The result survives the workbook's artifact warning.** Excluding every codebook-256 collision value leaves n = {len(sensitivity)} configurations and HR@10 ρ = **{hr_sensitivity:.3f}** ({_p_text(hr_sensitivity_p)}).

3. **The pooled relationship is not just a tokenizer-level contrast, but it is heterogeneous.** Within-tokenizer HR@10 associations are RQ-KMeans **{within_hr.loc['RQ-KMeans', 'spearman_rho']:+.3f}**, MQ **{within_hr.loc['MQ', 'spearman_rho']:+.3f}**, and RQ-VAE **{within_hr.loc['RQ-VAE', 'spearman_rho']:+.3f}**. RQ-VAE only spans {within_hr.loc['RQ-VAE', 'collision_min_pct']:.2f}–{within_hr.loc['RQ-VAE', 'collision_max_pct']:.2f}% collision, so it provides little range for this relationship.

4. **Collision ordering transfers across datasets.** Industrial versus Office collision has ρ = **{cross_rho:.3f}** ({_p_text(cross_p)}). The main exception is RQ-KMeans, 5 digits × 512: {outlier.Industrial:.4f}% on Industrial versus {outlier.Office:.4f}% on Office (**{outlier.office_to_industrial_ratio:.1f}×**).

5. **Longer SIDs expose a paradigm gap.** Mean diffusion-family minus autoregressive HR@10 is **{depth_gaps.loc[3]:+.2f}** at 3 digits, **{depth_gaps.loc[4]:+.2f}** at 4, and **{depth_gaps.loc[5]:+.2f}** at 5. All {len(five_digit)} five-digit cells are negative (two-sided exact Wilcoxon p = {five_wilcoxon.pvalue:.4f} across configuration cells).

6. **The six performance metrics are nearly redundant.** Their pairwise configuration-level Spearman correlations span **{upper.min():.3f}–{upper.max():.3f}**, supporting HR@10 and NDCG@10 as compact headline views.
{cross_dataset_finding}
{resource_finding}

## Figures

| Figure | Question answered |
|---|---|
{figure_rows}

Every Matplotlib figure is emitted as both a 400-dpi PNG and a vector PDF. Tidy source and derived tables are under `data/` beside this report.

## Interpretation limits

- These are **associations, not causal estimates** of collision's effect. Tokenizer, depth, and codebook jointly determine collision and also co-vary with reported performance.
- There is one value per configuration and no seed/run variance. Configuration-cell p-values describe ordering across design cells; they are not experimental error bars or training uncertainty.
- Industrial codebook-256 collision values are labeled “old artifact/from previous runs,” and every Office collision value is labeled as a previous run. Sensitivity results therefore exclude codebook 256 where possible.
- The analysis maps collision-table `MQ` to performance-table `Parallel SID`; this is inferred from the layout and naming.
- Rows labeled `Mask Diffusion` at codebooks 128/512 and `DiffGRM` at 256 are grouped as a **diffusion-family paradigm**. That grouping does not establish that the implementations are identical.
{cross_dataset_limit}{resource_limit}- Metric units appear to be percentage points, but the workbook does not explicitly state units.
- A synthetic “collision-corrected leaderboard” is intentionally omitted: the design variables are structurally collinear, so such counterfactual rankings would be assumption-heavy without item-level predictions or controlled reruns.
"""
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def generate_analysis(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    workbook: Path,
    office_runs: pd.DataFrame | None = None,
    research_notes: Path | None = None,
    resources: pd.DataFrame | None = None,
) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    derived = build_derived_tables(tables, office_runs=office_runs)
    prepared_resources: pd.DataFrame | None = None
    if resources is not None and not resources.empty:
        prepared_resources = prepare_resource_table(resources)
        derived["resource_summary"] = resource_summary(prepared_resources)
    write_derived_tables(derived, output_dir)

    fig01_collision_design_space(tables["collisions"], output_dir)
    fig02_cross_dataset_collision(derived["cross_dataset_collisions"], output_dir)
    fig03_collision_vs_reported_metrics(
        tables["core_runs"], derived["configuration_means"], output_dir
    )
    fig04_within_tokenizer_association(
        derived["configuration_means"],
        derived["within_tokenizer_associations"],
        output_dir,
    )
    fig05_metric_robustness(derived["association_summary"], output_dir)
    fig06_paradigm_gap_by_depth(derived["paradigm_gaps"], output_dir)
    fig07_model_training_ablation(
        derived["model_training_ablation_3d_codebook256"], output_dir
    )
    fig08_full_configuration_grid(
        derived["configuration_means"],
        derived["paradigm_gaps"],
        derived["cross_dataset_collisions"],
        output_dir,
    )
    fig09_factorial_main_effects(derived["factorial_effects"], output_dir)
    if "cross_dataset_paradigm_gaps_codebook256" in derived:
        fig10_cross_dataset_paradigm_gap(
            derived["cross_dataset_paradigm_gaps_codebook256"], output_dir
        )
    if prepared_resources is not None:
        fig11_training_compute_resources(prepared_resources, output_dir)
        fig12_sid_depth_tradeoff(
            tables["collisions"],
            derived["paradigm_gaps"],
            prepared_resources,
            output_dir,
        )
    write_report(
        tables,
        derived,
        output_dir,
        workbook,
        research_notes=research_notes,
        resources=prepared_resources,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--research-notes", type=Path, default=DEFAULT_PDF)
    parser.add_argument(
        "--resource-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "data" / "resource_runs.csv",
        help="optional resource snapshot produced by parse_resources.py",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    tables = build_tables(args.workbook.resolve())
    office_runs = build_office_table(args.research_notes.resolve())
    resources = (
        pd.read_csv(args.resource_csv.resolve()) if args.resource_csv.exists() else None
    )
    generate_analysis(
        tables,
        args.output_dir.resolve(),
        args.workbook.resolve(),
        office_runs=office_runs,
        research_notes=args.research_notes.resolve(),
        resources=resources,
    )


if __name__ == "__main__":
    main()
