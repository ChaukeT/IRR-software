from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QAbstractItemView,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ResultsPanel(QWidget):
    """Widget used to display optimisation results after a successful run."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._results_df: pd.DataFrame | None = None
        self._pid_column: str | None = None

        layout = QVBoxLayout(self)

        self.summary_label = QLabel("No optimisation run yet.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.block_model_table = QTableWidget()
        self.block_model_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.block_model_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.block_model_table, 2)

        export_row = QHBoxLayout()
        export_row.addStretch(1)
        self.export_button = QPushButton("Export optimised block model…")
        self.export_button.clicked.connect(self._export_results)
        self.export_button.setEnabled(False)
        export_row.addWidget(self.export_button)
        layout.addLayout(export_row)

        self.irr_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self.irr_canvas, 2)

        self.period_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self.period_canvas, 2)

        self.other_results = QTextEdit()
        self.other_results.setReadOnly(True)
        self.other_results.setPlaceholderText("Additional mine metrics will appear here once available.")
        layout.addWidget(self.other_results, 1)

        self.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset the panel to its initial, empty state."""

        self.summary_label.setText("No optimisation run yet.")
        self.block_model_table.clear()
        self.block_model_table.setRowCount(0)
        self.block_model_table.setColumnCount(0)
        self.block_model_table.setHorizontalHeaderLabels([])
        self.export_button.setEnabled(False)
        self.other_results.clear()
        self.other_results.setPlainText("")

        self.irr_canvas.figure.clear()
        self.irr_canvas.draw()
        self.period_canvas.figure.clear()
        self.period_canvas.draw()

        self._results_df = None
        self._pid_column = None

    def update_results(
        self,
        *,
        r_star: float,
        schedule: dict[str, int],
        block_df: pd.DataFrame | None,
        iteration_history: Iterable[tuple[int, float, bool]],
        extra_info: dict[str, Any],
    ) -> None:
        """Populate the panel with fresh optimisation output."""

        self.summary_label.setText(f"Optimised IRR: {r_star:.4%}")
        self._results_df = self._build_results_dataframe(block_df, schedule, r_star)
        note = self._populate_table(self._results_df)
        self._plot_irr_history(list(iteration_history), r_star)
        self._plot_period_profile(block_df, schedule)
        self._populate_other_results(extra_info, schedule, r_star, note)

        self.export_button.setEnabled(self._results_df is not None and not self._results_df.empty)

    def show_error(self, message: str) -> None:
        """Display an error message in the panel."""

        self.clear()
        self.summary_label.setText(message)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_results_dataframe(
        self,
        block_df: pd.DataFrame | None,
        schedule: dict[str, int],
        r_star: float,
    ) -> pd.DataFrame:
        if block_df is None or block_df.empty:
            data = [
                {"Panel": pid, "Optimised_Period": period, "Optimised_IRR": r_star}
                for pid, period in sorted(schedule.items(), key=lambda kv: (kv[1], kv[0]))
            ]
            return pd.DataFrame(data)

        result_df = block_df.copy()
        pid_column = self._detect_pid_column(result_df)
        self._pid_column = pid_column

        if pid_column is None:
            data = [
                {"Panel": pid, "Optimised_Period": period, "Optimised_IRR": r_star}
                for pid, period in sorted(schedule.items(), key=lambda kv: (kv[1], kv[0]))
            ]
            return pd.DataFrame(data)

        result_df["Optimised_Period"] = result_df[pid_column].map(schedule)
        result_df["Optimised_IRR"] = r_star
        return result_df

    def _populate_table(self, df: pd.DataFrame) -> str | None:
        self.block_model_table.setRowCount(0)
        self.block_model_table.setColumnCount(0)

        if df.empty:
            self.block_model_table.setHorizontalHeaderLabels([])
            return None

        display_columns: list[str] = []
        if self._pid_column and self._pid_column in df.columns:
            display_columns.append(self._pid_column)
        if "Optimised_Period" in df.columns:
            display_columns.append("Optimised_Period")
        if "Optimised_IRR" in df.columns:
            display_columns.append("Optimised_IRR")

        if not display_columns:
            display_columns = list(df.columns[: min(6, len(df.columns))])

        preview_df = df.loc[:, display_columns].copy()
        max_rows = min(len(preview_df), 500)
        preview_df = preview_df.head(max_rows)

        self.block_model_table.setColumnCount(len(display_columns))
        self.block_model_table.setHorizontalHeaderLabels(display_columns)
        self.block_model_table.setRowCount(len(preview_df))

        for row_idx, (_, row) in enumerate(preview_df.iterrows()):
            for col_idx, column in enumerate(display_columns):
                value = row[column]
                text = "" if pd.isna(value) else str(value)
                self.block_model_table.setItem(row_idx, col_idx, QTableWidgetItem(text))

        self.block_model_table.resizeColumnsToContents()
        if len(df) > max_rows:
            return f"Table preview truncated to the first {max_rows} rows out of {len(df)} total."
        return None

    def _plot_irr_history(self, history: list[tuple[int, float, bool]], r_star: float) -> None:
        fig = self.irr_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        if not history:
            ax.set_title("IRR search progression")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("IRR")
            ax.text(0.5, 0.5, "No progress information", ha="center", va="center", transform=ax.transAxes)
        else:
            iterations = [it for it, _, _ in history]
            values = [r for _, r, _ in history]
            feasibility = [feasible for _, _, feasible in history]

            colors = ["tab:green" if feas else "tab:red" for feas in feasibility]
            ax.plot(iterations, values, color="tab:blue", alpha=0.3)
            ax.scatter(iterations, values, c=colors, edgecolor="black", linewidths=0.5)
            ax.axhline(r_star, color="tab:purple", linestyle="--", label=f"Optimised IRR = {r_star:.4%}")
            ax.set_title("IRR search progression")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("IRR")
            ax.legend(loc="best")

        fig.tight_layout()
        self.irr_canvas.draw()

    def _plot_period_profile(
        self,
        block_df: pd.DataFrame | None,
        schedule: dict[str, int],
    ) -> None:
        fig = self.period_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        if not schedule:
            ax.set_title("Period allocation profile")
            ax.text(0.5, 0.5, "No schedule information", ha="center", va="center", transform=ax.transAxes)
            fig.tight_layout()
            self.period_canvas.draw()
            return

        period_totals = self._compute_period_totals(block_df, schedule)
        if isinstance(period_totals, pd.Series):
            periods = period_totals.index.tolist()
            values = period_totals.to_list()
            ylabel = period_totals.name or "Total"
        else:
            periods = list(period_totals.keys())
            values = list(period_totals.values())
            ylabel = "Total"

        if not periods:
            ax.set_title("Period allocation profile")
            ax.text(0.5, 0.5, "No assigned periods", ha="center", va="center", transform=ax.transAxes)
            fig.tight_layout()
            self.period_canvas.draw()
            return

        ax.bar(periods, values, color="tab:orange")
        ax.set_title("IRR allocation by period")
        ax.set_xlabel("Period")
        ax.set_ylabel(ylabel)
        ax.set_xticks(periods)
        ax.set_xticklabels([str(p) for p in periods])

        fig.tight_layout()
        self.period_canvas.draw()

    def _populate_other_results(
        self,
        extra_info: dict[str, Any],
        schedule: dict[str, int],
        r_star: float,
        table_note: str | None,
    ) -> None:
        lines: list[str] = []

        if "npv" in extra_info:
            try:
                lines.append(f"Net present value (NPV): {float(extra_info['npv']):,.2f}")
            except (TypeError, ValueError):
                lines.append(f"Net present value (NPV): {extra_info['npv']}")

        if "status" in extra_info:
            lines.append(f"Solver status code: {extra_info['status']}")

        if "dir" in extra_info:
            lines.append(f"Run directory: {extra_info['dir']}")

        if "summary" in extra_info and isinstance(extra_info["summary"], dict):
            for key, value in extra_info["summary"].items():
                lines.append(f"{key}: {value}")

        if self._results_df is not None and not self._results_df.empty:
            total_panels = len(self._results_df)
            scheduled_panels = sum(1 for period in schedule.values() if period is not None)
            lines.append(
                f"Scheduled panels: {scheduled_panels} out of {total_panels} (IRR {r_star:.2%})."
            )

        if not lines:
            lines.append("No additional metrics were provided by the solver.")

        if table_note:
            lines.append(table_note)

        self.other_results.setPlainText("\n".join(lines))

    def _compute_period_totals(
        self,
        block_df: pd.DataFrame | None,
        schedule: dict[str, int],
    ) -> pd.Series:
        if not schedule:
            return pd.Series(dtype=float)

        period_counts = Counter(schedule.values())
        totals = pd.Series(period_counts).sort_index()
        totals.name = "Panels"

        if block_df is None or self._pid_column is None or self._pid_column not in block_df.columns:
            return totals

        tonnage_column = self._detect_tonnage_column(block_df)
        if tonnage_column is None:
            return totals

        df = block_df[[self._pid_column, tonnage_column]].copy()
        df["Optimised_Period"] = df[self._pid_column].map(schedule)
        df = df.dropna(subset=["Optimised_Period"])
        if df.empty:
            return totals

        agg = df.groupby("Optimised_Period")[tonnage_column].sum().sort_index()
        agg.name = tonnage_column
        return agg

    def _detect_pid_column(self, df: pd.DataFrame) -> str | None:
        preferred = ["pid", "panel", "panel_id", "panelid", "block", "block_id"]
        for column in df.columns:
            col_lower = column.lower()
            if col_lower in preferred or ("panel" in col_lower and "id" in col_lower):
                return column
        return None

    def _detect_tonnage_column(self, df: pd.DataFrame) -> str | None:
        preferred = {"tonnes", "tonnage", "tons", "mass", "ore_tonnes"}
        for column in df.columns:
            if column.lower() in preferred:
                return column
        return None

    def _export_results(self) -> None:
        if self._results_df is None or self._results_df.empty:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save optimised block model",
            str(Path.home() / "optimised_block_model.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        try:
            self._results_df.to_csv(path, index=False)
            self.other_results.append(f"Saved optimised block model to {path}.")
        except Exception as exc:  # pragma: no cover - GUI feedback path
            self.other_results.append(f"Failed to save results: {exc}")
