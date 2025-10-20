from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from engine_adapter import EngineAdapter
from io import load_blockmodel_csv
from progress_log import ProgressLog
from schedule_table import ScheduleTable


class MainWindow(QWidget):
    """Main application window tying together the simple PySide6 UI."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("IRR Scheduler")

        self._engine = EngineAdapter()
        self._panels: Any = None
        self._arcs: Any = None
        self._blockmodel_path: Path | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._btn_load = QPushButton("Load blockmodel…")
        self._btn_load.clicked.connect(self.on_load)

        self._btn_run = QPushButton("Run IRR search")
        self._btn_run.clicked.connect(self.on_run)
        self._btn_run.setEnabled(False)

        self._backend = QComboBox()
        self._backend.addItems(["local", "smps"])

        self._spin_T = QSpinBox()
        self._spin_T.setRange(1, 200)
        self._spin_T.setValue(20)

        self._rom_cap = QDoubleSpinBox()
        self._rom_cap.setRange(0.0, 1_000_000.0)
        self._rom_cap.setDecimals(0)
        self._rom_cap.setSingleStep(1_000.0)
        self._rom_cap.setValue(50_000.0)

        self._r_lo = QDoubleSpinBox()
        self._r_lo.setRange(-1.0, 1.0)
        self._r_lo.setSingleStep(0.01)
        self._r_lo.setValue(0.0)

        self._r_hi = QDoubleSpinBox()
        self._r_hi.setRange(-1.0, 5.0)
        self._r_hi.setSingleStep(0.01)
        self._r_hi.setValue(0.3)

        self._tol = QDoubleSpinBox()
        self._tol.setDecimals(4)
        self._tol.setRange(0.0001, 1.0)
        self._tol.setSingleStep(0.0005)
        self._tol.setValue(0.0005)

        left_box = QGroupBox("Controls")
        left_layout = QFormLayout(left_box)
        left_layout.addRow(self._btn_load)
        left_layout.addRow("Backend", self._backend)
        left_layout.addRow("Horizon (T)", self._spin_T)
        left_layout.addRow("ROM capacity", self._rom_cap)
        left_layout.addRow("IRR lower bound", self._r_lo)
        left_layout.addRow("IRR upper bound", self._r_hi)
        left_layout.addRow("Tolerance", self._tol)
        left_layout.addRow(self._btn_run)

        self._lbl_status = QLabel("No run yet")
        self._log = ProgressLog()
        self._table = ScheduleTable()

        right_box = QGroupBox("Results")
        right_layout = QVBoxLayout(right_box)
        right_layout.addWidget(self._lbl_status)
        right_layout.addWidget(self._log, 2)
        right_layout.addWidget(self._table, 3)

        root = QVBoxLayout(self)
        root.addWidget(left_box, alignment=Qt.AlignTop)
        root.addWidget(right_box, stretch=1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def on_load(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select blockmodel CSV", "", "CSV files (*.csv)"
        )
        if not path_str:
            return

        path = Path(path_str)
        try:
            panels, idx_map, meta, rock_by = load_blockmodel_csv(path)
            from irr_core.modelio import build_slope_arcs

            arcs = build_slope_arcs(idx_map, meta, per_material=True, rock_by=rock_by)
        except Exception as exc:  # pragma: no cover - defensive GUI path
            QMessageBox.critical(self, "Load error", str(exc))
            return

        self._panels = panels
        self._arcs = arcs
        self._blockmodel_path = path
        self._btn_run.setEnabled(True)
        self._log.append(
            f"Loaded {len(panels)} panels; built {len(arcs)} arcs from {path.name}"
        )

    def on_run(self) -> None:
        if not self._panels or not self._arcs:
            QMessageBox.warning(self, "No data", "Load a blockmodel first")
            return

        from irr_core.econ import Econ

        econ = Econ(
            recovery=0.9,
            au_price_g=1.2,
            cu_price_t=9000.0,
            mining_cost_ore=40.0,
            processing_cost=15.0,
            waste_costs={"Soil": 15.0, "Weathered": 20.0, "Fresh": 25.0},
        )

        self._log.clear()
        self._lbl_status.setText("Running…")
        info = self._engine.run(
            panels=self._panels,
            arcs=self._arcs,
            econ=econ,
            backend=self._backend.currentText(),
            T=self._spin_T.value(),
            ROM_cap=float(self._rom_cap.value()),
            r_lo=float(self._r_lo.value()),
            r_hi=float(self._r_hi.value()),
            tol=float(self._tol.value()),
            progress_cb=self._log.progress_cb,
        )

        if info.get("ok"):
            r_star = info.get("r", 0.0)
            self._lbl_status.setText(f"IRR*: {r_star:.4f}")
            sched = info.get("schedule", {})
            self._table.set_schedule(sched)
            self._log.append("Done.")
        else:
            self._lbl_status.setText("Run failed")
            self._log.append(str(info.get("error", "Unknown error")))
