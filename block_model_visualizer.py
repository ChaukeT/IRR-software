"""Block model visualization utilities using PyVista.

This module reads a structured block model exported as a CSV file and rebuilds
it as a PyVista ``UniformGrid`` for interactive 3D exploration.  It offers
optional filtering, configurable property visualization, and export helpers for
VTK datasets and rendered screenshots.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyvista as pv

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "I",
    "J",
    "K",
    "XC",
    "YC",
    "ZC",
    "ZONE",
    "XINC",
    "YINC",
    "ZINC",
    "XMORIG",
    "YMORIG",
    "ZMORIG",
    "NX",
    "NY",
    "NZ",
    "Au",
    "Cu",
    "LITO",
    "DENSIT",
)


def load_csv_data(csv_path: Path | str) -> pd.DataFrame:
    """Load block model information from ``csv_path`` and validate columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV file is missing required columns: " + ", ".join(sorted(missing))
        )

    return df


def _sort_for_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe sorted in a consistent I-J-K order."""
    sort_cols = [col for col in ("K", "J", "I") if col in df.columns]
    if not sort_cols:
        return df.copy()
    # A stable sort preserves the original order within planes.
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def create_voxel_grid(
    df: pd.DataFrame,
) -> Tuple[pv.UniformGrid, pd.DataFrame, Dict[str, Dict[float, str]]]:
    """Create a ``UniformGrid`` from a block-model dataframe.

    Returns the grid, the sorted dataframe used for ordering, and optional
    categorical mappings for non-numeric columns stored as cell data.
    """

    if df.empty:
        raise ValueError("The provided dataframe is empty; nothing to visualize.")

    df_sorted = _sort_for_grid(df)

    origin = (
        float(df_sorted["XMORIG"].iloc[0]),
        float(df_sorted["YMORIG"].iloc[0]),
        float(df_sorted["ZMORIG"].iloc[0]),
    )
    spacing = (
        float(df_sorted["XINC"].iloc[0]),
        float(df_sorted["YINC"].iloc[0]),
        float(df_sorted["ZINC"].iloc[0]),
    )
    dimensions = (
        int(df_sorted["NX"].iloc[0]),
        int(df_sorted["NY"].iloc[0]),
        int(df_sorted["NZ"].iloc[0]),
    )

    expected_cells = int(np.prod(dimensions))
    if expected_cells != len(df_sorted):
        raise ValueError(
            "Block count does not match grid dimensions: "
            f"expected {expected_cells} cells but found {len(df_sorted)} rows"
        )

    grid = pv.UniformGrid()
    grid.dimensions = np.array(dimensions) + 1  # UniformGrid expects point dims.
    grid.origin = origin
    grid.spacing = spacing

    # Add frequently used coordinate/cell metadata arrays.
    grid.cell_data["XC"] = df_sorted["XC"].to_numpy(dtype=float, copy=False)
    grid.cell_data["YC"] = df_sorted["YC"].to_numpy(dtype=float, copy=False)
    grid.cell_data["ZC"] = df_sorted["ZC"].to_numpy(dtype=float, copy=False)

    categorical_maps: Dict[str, Dict[float, str]] = {}
    metadata_columns = set(REQUIRED_COLUMNS) | {"XC", "YC", "ZC"}

    for column in df_sorted.columns:
        if column in metadata_columns:
            continue
        series = df_sorted[column]
        if pd.api.types.is_numeric_dtype(series):
            grid.cell_data[column] = series.to_numpy(dtype=float, copy=False)
        else:
            categories = pd.Categorical(series)
            codes = categories.codes.astype(float)
            codes[codes < 0] = np.nan
            grid.cell_data[column] = codes
            categorical_maps[column] = {
                float(code): str(label) for code, label in enumerate(categories.categories)
            }

    # Also store the required properties even if they were skipped above.
    for column in ("Au", "Cu", "LITO", "DENSIT", "ZONE"):
        if column in grid.cell_data:
            continue
        series = df_sorted[column]
        if pd.api.types.is_numeric_dtype(series):
            grid.cell_data[column] = series.to_numpy(dtype=float, copy=False)
        else:
            categories = pd.Categorical(series)
            codes = categories.codes.astype(float)
            codes[codes < 0] = np.nan
            grid.cell_data[column] = codes
            categorical_maps[column] = {
                float(code): str(label) for code, label in enumerate(categories.categories)
            }

    return grid, df_sorted, categorical_maps


def apply_property_colors(
    grid: pv.UniformGrid,
    df_sorted: pd.DataFrame,
    property_name: str,
    mask: Optional[Sequence[bool]] = None,
    categorical_maps: Optional[Dict[str, Dict[float, str]]] = None,
) -> Dict[float, str]:
    """Populate ``grid`` with the selected property and return annotations."""
    if property_name not in df_sorted.columns and property_name not in grid.cell_data:
        raise KeyError(f"Property '{property_name}' not present in the dataset")

    if property_name in df_sorted.columns:
        property_series = df_sorted[property_name]
    else:
        property_series = pd.Series(grid.cell_data[property_name])

    if mask is not None:
        if len(mask) != len(property_series):
            raise ValueError("Mask length does not match number of cells")
        property_series = property_series.mask(~np.asarray(mask))

    annotations: Dict[float, str] = {}
    if pd.api.types.is_numeric_dtype(property_series.dropna()):
        grid.cell_data[property_name] = property_series.to_numpy(dtype=float)
    else:
        categories = pd.Categorical(property_series)
        codes = categories.codes.astype(float)
        codes[codes < 0] = np.nan
        grid.cell_data[property_name] = codes
        if categorical_maps and property_name in categorical_maps:
            annotations = categorical_maps[property_name]
        else:
            annotations = {
                float(code): str(label)
                for code, label in enumerate(categories.categories)
            }

    return annotations


def visualize_model(
    grid: pv.UniformGrid,
    property_name: str,
    annotations: Optional[Dict[float, str]] = None,
    cmap: str = "viridis",
) -> pv.Plotter:
    """Create a PyVista plotter visualizing ``property_name`` on the grid."""
    plotter = pv.Plotter()

    scalar_args = dict(
        scalars=property_name,
        cmap=cmap,
        show_edges=False,
        opacity=1.0,
        nan_opacity=0.05,
        scalar_bar_args={"title": property_name},
    )
    if annotations:
        scalar_args.update({"categories": True, "annotations": annotations})

    plotter.add_mesh(grid, **scalar_args)
    plotter.add_axes(line_width=2)
    plotter.show_bounds(grid="back", location="outer", all_edges=True)
    plotter.set_background("#1e1e1e")
    plotter.camera_position = "iso"
    plotter.enable_eye_dome_lighting()
    plotter.add_title(f"Block Model — {property_name}")

    return plotter


def export_model(grid: pv.UniformGrid, output_path: Path | str) -> Path:
    """Export the grid to ``output_path`` as .vti or .vtm."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".vtm":
        multiblock = pv.MultiBlock({"block_model": grid})
        multiblock.save(output_path)
    elif suffix in {".vti", ".vtk"}:
        grid.save(output_path)
    else:
        raise ValueError("Unsupported export format. Use .vtm, .vti, or .vtk")

    return output_path


def save_screenshot(plotter: pv.Plotter, output_path: Path | str) -> Path:
    """Render the current scene to ``output_path`` as a PNG image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path), transparent_background=False)
    return output_path


def _parse_zone_filter(values: Optional[Sequence[str]]) -> Optional[Sequence[str]]:
    if not values:
        return None
    # Expand comma-separated values provided as a single string.
    if len(values) == 1 and "," in values[0]:
        return [val.strip() for val in values[0].split(",") if val.strip()]
    return values


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize a block model CSV with PyVista")
    parser.add_argument("csv", type=Path, help="Path to the block model CSV file")
    parser.add_argument(
        "--property",
        dest="property_name",
        default="Cu",
        help="Name of the property column to visualize (default: Cu)",
    )
    parser.add_argument(
        "--filter-column",
        default="ZONE",
        help="Column used for categorical filtering (default: ZONE)",
    )
    parser.add_argument(
        "--filter-values",
        nargs="*",
        default=None,
        help="Allowed values for the filter column (e.g. --filter-values ZONE_A ZONE_B)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional output path for the grid (.vtm, .vti, or .vtk)",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Optional PNG path to save a rendered screenshot",
    )
    parser.add_argument(
        "--off-screen",
        action="store_true",
        help="Enable off-screen rendering (useful for headless environments)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the interactive viewer",
    )

    args = parser.parse_args(argv)

    if args.off_screen:
        pv.OFF_SCREEN = True

    df = load_csv_data(args.csv)
    grid, df_sorted, categorical_maps = create_voxel_grid(df)

    zone_values = _parse_zone_filter(args.filter_values)
    mask = None
    if zone_values:
        filter_column = args.filter_column
        if filter_column not in df_sorted.columns:
            raise KeyError(
                f"Filter column '{filter_column}' not present in the dataset"
            )
        mask = df_sorted[filter_column].isin(zone_values)

    annotations = apply_property_colors(
        grid,
        df_sorted,
        args.property_name,
        mask=mask,
        categorical_maps=categorical_maps,
    )
    plotter = visualize_model(grid, args.property_name, annotations=annotations)

    if args.export:
        export_model(grid, args.export)
    if args.screenshot:
        save_screenshot(plotter, args.screenshot)

    if not args.no_show:
        plotter.show()
    plotter.close()


if __name__ == "__main__":
    main()
